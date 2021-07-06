import collections
import math
import multiprocessing

from scipy import signal, fftpack
import numpy
import pygame
import pyaudio


def linearInterpolation(y1, y2, frac):
	return (y1 * (1.0 - frac) + y2 * frac)


def ampToDB(amp):
	if amp < 0.00001:
		return -200
	return 20.0 * math.log10(amp)


def DBToAmp(db):
	if db == 0:
		return 1
	return math.pow(10.0, db / 20)
	
def sineWave(sampleRate, channels, size):
	waveform = numpy.asarray([0.0] * size)
	frequency = sampleRate / size
	time = 0
	phase = 0
	deltaTime = 1.0 / sampleRate
	for i in range(0, size, channels):
		value = numpy.sin((2 * numpy.pi) * (frequency * (time + phase)))
		for j in range(channels):
			waveform[i + j] = value
		
		time += deltaTime
	return waveform


def squareWave(sampleRate, channels, size):
	waveform = [0.0] * size
	frequency = sampleRate / size
	time = 0
	phase = 0
	deltaTime = 1.0 / sampleRate
	for i in range(0, size, channels):
		value = numpy.sin((2 * numpy.pi) * (frequency * (time + phase)))
		if value >= 0:
			value = 1.0
		else:
			value = -1.0
		for j in range(channels):
			waveform[i + j] = value
		
		time += deltaTime
	return waveform

class ADSR():
	
	def __init__(self, sampleRate, channels):
		self.sampleRate = sampleRate
		self.channels = channels
		
		self.attackTime = int(0.1 * self.sampleRate)
		self.decayTime = int(0.1 * self.sampleRate)
		self.sustainValue = 0
		self.releaseTime = int(0.5 * self.sampleRate)
		self.releaseValue = 1
		
		self.currentTime = 0
		self.currentValue = 0
		
		self.noteState = False
		self.activeState = False
	
	def render(self, samplesToRender):
		envelope = numpy.zeros(samplesToRender)
		if self.activeState:
			for i in range(0, samplesToRender, self.channels):
				if self.noteState:
					if self.currentTime <= self.attackTime:
						self.currentValue = numpy.interp(self.currentTime, [0, self.attackTime], [0, 1])
						for j in range(self.channels):
							envelope[i + j] = self.currentValue
						self.currentTime += 1
					elif self.currentTime <= (self.attackTime + self.decayTime):
						self.currentValue = numpy.interp(self.currentTime, [self.attackTime, self.attackTime + self.decayTime], [1, self.sustainValue])
						for j in range(self.channels):
							envelope[i + j] = self.currentValue
						self.currentTime += 1
					else:
						if self.sustainValue != 0:
							self.currentValue = self.sustainValue
							for j in range(self.channels):
								envelope[i + j] = self.currentValue
						else:
							self.activeState = False
							break
				elif self.currentTime <= self.attackTime + self.decayTime + self.releaseTime:
					self.currentValue = numpy.interp(self.currentTime, [self.attackTime + self.decayTime, self.attackTime + self.decayTime + self.releaseTime], [self.releaseValue, 0])
					for j in range(self.channels):
						envelope[i + j] = self.currentValue
					self.currentTime += 1
				else:
					self.activeState = False
					break
		
		return envelope
	
	def noteOn(self):
		self.noteState = True
		self.activeState = True
		self.currentTime = 0
		self.currentValue = 0
	
	def noteOff(self):
		self.noteState = False
		self.currentTime = self.attackTime + self.decayTime
		self.releaseValue = self.currentValue

class LFO():
	def __init__(self, sampleRate, channels):
		self.sampleRate = sampleRate
		self.channels = channels
		
		self.frequency = 1
		self.amplitude = 1
		self.phase = 0.0
		self.phaseInSamples = 0
		self.amplitudeShift = 0.0
		
		self.waveTableLength = 1024
		self.waveTable = numpy.zeros(self.waveTableLength + 1)
		self.waveTable[: self.waveTableLength] = sineWave(self.sampleRate,1,self.waveTableLength)
		self.baseFrequency = self.sampleRate / self.waveTableLength
		self.readSpeed = self.frequency / self.baseFrequency
		self.readIndex = 0.0
		
	def render(self, samplesToRender):
		waveform = numpy.empty(samplesToRender)
		
		for i in range(0, samplesToRender, self.channels):
			readIndexRoundDown = int(self.readIndex + self.phaseInSamples) % self.waveTableLength
			readIndexRoundUp = int(self.readIndex + self.phaseInSamples + 1) % self.waveTableLength
			readIndexFractional = self.readIndex - int(self.readIndex)
			
			value = linearInterpolation(self.waveTable[readIndexRoundDown], self.waveTable[readIndexRoundUp], readIndexFractional)
			for j in range(self.channels):
				waveform[i + j] = (value * self.amplitude) + self.amplitudeShift
			
			self.readIndex += self.readSpeed
		
		return waveform
	

	def setWavetable(self, waveTable):
		self.waveTableLength = len(waveTable)
		self.waveTable = numpy.zeros(self.waveTableLength + 1)
		
		# sig_fft = fftpack.fft(waveTable)
		# freq = fftpack.fftfreq(sig_fft.size, d = 1 / self.sampleRate)
		# sig_fft = sig_fft
		# for i in range(len(freq)):
		# 	if abs(freq[i]) > 18000:
		# 		sig_fft[i] = 0
		#
		# waveTable = fftpack.ifft(sig_fft)
		# waveTable = waveTable.real
		
		maxValue = max(abs(waveTable))
		if maxValue != 0:
			scale = (1.0 / maxValue)
			waveTable = scale * waveTable
		
		self.waveTable[: self.waveTableLength] = waveTable
		self.baseFrequency = self.sampleRate / self.waveTableLength
		self.readSpeed = self.frequency / self.baseFrequency
		
	def setPhase(self, phase):
		self.phase = phase
		self.phaseInSamples = (self.waveTableLength / 360) * phase
		
	def setFrequency(self, frequency):
		self.frequency = frequency
		self.readSpeed = self.frequency / self.baseFrequency
		self.readIndex = 0.0
		
	def setAmplitude(self, amplitude):
		self.amplitude = amplitude
	
	def setAmplitudeShift(self, shift):
		self.amplitudeShift = shift

class synthesizer():
	
	def __init__(self, sampleRate, channels):
		self.sampleRate = sampleRate
		self.channels = channels
		
		self.frequency = 440.0
		self.amplitude = 0.9
		self.phase = 0.0
		
		self.time = 0.0
		self.deltaTime = 1.0 / self.sampleRate
		
		self.waveTableLength = 1024
		self.waveTable = numpy.zeros(self.waveTableLength + 1)
		self.waveTable[: self.waveTableLength] = 0.0
		self.baseFrequency = self.sampleRate / self.waveTableLength
		self.readSpeed = self.frequency / self.baseFrequency
		self.readIndex = 0.0
		
		self.ADSR = ADSR(self.sampleRate, self.channels)
	
	def render(self, samplesToRender):
		waveform = numpy.empty(samplesToRender)
		
		for i in range(0, samplesToRender, self.channels):
			readIndexRoundDown = int(self.readIndex) % self.waveTableLength
			readIndexRoundUp = int(self.readIndex + 1) % self.waveTableLength
			readIndexFractional = self.readIndex - int(self.readIndex)
			
			value = linearInterpolation(self.waveTable[readIndexRoundDown], self.waveTable[readIndexRoundUp], readIndexFractional)
			for j in range(self.channels):
				waveform[i + j] = value * self.amplitude
			
			self.readIndex += self.readSpeed
		
		envelope = self.ADSR.render(samplesToRender)
		waveform *= envelope
		return waveform
	
	def setWavetable(self, waveTable):
		sig_fft = fftpack.fft(waveTable)
		freq = fftpack.fftfreq(sig_fft.size, d = 1 / 44100)
		sig_fft = sig_fft
		for i in range(len(freq)):
			if abs(freq[i]) > 16000:
				sig_fft[i] = 0
		
		waveTable = fftpack.ifft(sig_fft)
		waveTable = waveTable.real
		
		maxValue = max(abs(waveTable))
		if maxValue != 0:
			scale = (1.0 / maxValue)
			waveTable = scale * waveTable
		
		self.waveTable[: self.waveTableLength] = waveTable
		self.baseFrequency = self.sampleRate / self.waveTableLength
		self.readSpeed = self.frequency / self.baseFrequency
	
	def noteIn(self, frequency):
		if self.frequency == frequency and self.ADSR.activeState:
			self.noteOff(frequency)
		else:
			self.noteOn(frequency)
	
	def noteOn(self, frequency):
		self.frequency = frequency
		self.readSpeed = self.frequency / self.baseFrequency
		self.ADSR.noteOn()
	
	def noteOff(self, frequency):
		self.ADSR.noteOff()


class audioHandler(multiprocessing.Process):
	
	def __init__(self):
		super(multiprocessing.Process, self).__init__()
		self.daemon = True
		
		self.callbackPipeChild, self.callbackPipeParent = multiprocessing.Pipe(duplex = False)
		self.samplesQueue = multiprocessing.Queue()
		self.messageQueue = multiprocessing.Queue()
		
		self.bitDepth = 32
		self.sampleRate = 44100
		self.channels = 2
		self.frameCount = self.sampleRate // 32
		self.bufferSize = self.frameCount * self.channels
		
		self.running = False
	
	def audioCallback(self, in_data, frame_count, time_info, status):
		if status == pyaudio.paOutputOverflow or status == pyaudio.paOutputUnderflow:
			print("Underflow / Overflow")
		
		samples = self.callbackPipeChild.recv()
		self.samplesQueue.put_nowait(samples)
		
		return (numpy.array(samples, dtype = numpy.float32), pyaudio.paContinue)
	
	def input(self):
		while not self.messageQueue.empty():
			message = self.messageQueue.get_nowait()
			messageKey = message[0]
			messageValue = message[1]
			
			if messageKey == "STOP":
				self.running = False
	
	def run(self):
		pyAudio = pyaudio.PyAudio()
		audioCallback = pyAudio.open(format = pyaudio.paFloat32, channels = self.channels, rate = self.sampleRate, frames_per_buffer = self.frameCount, stream_callback = self.audioCallback, output = True, start = False)
		audioCallback.start_stream()
		
		self.running = True
		while self.running:
			samples = numpy.zeros(self.bufferSize)
			self.callbackPipeParent.send(samples)  # Blocking if size == 1
			self.input()
		
		audioCallback.stop_stream()
		audioCallback.close()
		pyAudio.terminate()
		self.terminate()
	
	def stop(self):
		self.messageQueue.put_nowait(["STOP", None])
	
class graphicHandler:
	
	def __init__(self, audioHandler):
		pygame.init()
		pygame.display.set_caption("Lissajous")
		
		self.clock = pygame.time.Clock()
		self.width = 720
		self.height = 720
		self.screen = pygame.display.set_mode((self.width, self.height))
		self.deltaTime = 0
		self.running = False
		
		self.audioHandler = audioHandler
		self.samples = collections.deque(maxlen = int((self.audioHandler.bufferSize)))
		self.samples.extend([0.0] * int((self.audioHandler.bufferSize)))
		
		self.LFO1 = LFO(self.audioHandler.sampleRate, 1)
		self.LFO2 = LFO(self.audioHandler.sampleRate, 1)
		self.LFO1.setFrequency(2)
		self.LFO2.setPhase(90)
		self.LFO2.setFrequency(2)
		
		self.lissajous = lissajousDrawer(self.width // 2, self.height // 2, 0, 0, self.LFO1, self.LFO2)
		self.wavetablePainter = wavetablePainter(self.width // 2, self.height // 2, 0, self.height // 2)
	
	def start(self):
		self.audioHandler.start()
		self.render()
	
	def stop(self):
		self.running = False
	
	def getAudioSamplesForFrame(self):
		while not self.audioHandler.samplesQueue.empty():
			self.samples.extend(self.audioHandler.samplesQueue.get_nowait())
		
		samplesToRender = int(self.audioHandler.sampleRate * self.deltaTime)
		samples = []
		for i in range(samplesToRender):
			if self.samples:
				samples.append(self.samples.popleft())
			else:
				samples.append(0.0)
		
		return samples
	
	def render(self):
		self.running = True
		while self.running:
			self.screen.fill((0, 0, 0), [0, 360, 360, 360])
			self.deltaTime = 0.016 if self.deltaTime == 0.0 else self.clock.get_time() / 1000
			
			font = pygame.font.SysFont('Times New Roman', 24)
			textsurface = font.render('LFO1 Frequency: ' + str(self.LFO1.frequency)[:3], False, (255, 255, 255))
			self.screen.blit(textsurface, (self.width // 2, self.height - 28))
			textsurface = font.render('LFO2 Frequency: ' + str(self.LFO2.frequency)[:3], False, (255, 255, 255))
			self.screen.blit(textsurface, (self.width // 2, self.height - 28 * 2))
			textsurface = font.render('LFO1 Phase: ' + str(self.LFO1.phase)[:3], False, (255, 255, 255))
			self.screen.blit(textsurface, (self.width // 2, self.height - 28 * 3))
			textsurface = font.render('LFO2 Phase: ' + str(self.LFO2.phase)[:3], False, (255, 255, 255))
			self.screen.blit(textsurface, (self.width // 2, self.height - 28 * 4))
			
			samples = self.getAudioSamplesForFrame()
			
			points = self.lissajous.render(len(samples))
			
			for point in points[0]:
				pygame.draw.circle(self.screen, [255,255,255], point, 2)
			for point in points[1]:
				pygame.draw.circle(self.screen, [255,150,150], [int(point), 0], 2)
			for point in points[2]:
				pygame.draw.circle(self.screen, [255,150,150], [0, int(point)], 2)
				
			points = self.wavetablePainter.render()
			for i in range(self.wavetablePainter.width - 1):
				pygame.draw.line(self.screen, [255,255,255,255], [i, points[i]], [i+1, points[i+1]])

			pygame.display.flip()
			self.clock.tick(60)
			self.input()
	
	def input(self):
		events = pygame.event.get()
		for event in events:
			if event.type == pygame.QUIT:
				self.audioHandler.stop()
				self.stop()
				pygame.quit()
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_v:
					self.LFO1.setWavetable(self.wavetablePainter.updateWaveTable())
					self.LFO2.setWavetable(self.wavetablePainter.updateWaveTable())
				if event.key == pygame.K_z:
					self.wavetablePainter.smoothPixelTable()
					self.LFO1.setWavetable(self.wavetablePainter.updateWaveTable())
					self.LFO2.setWavetable(self.wavetablePainter.updateWaveTable())
				if event.key == pygame.K_b:
					self.LFO1.setWavetable(sineWave(self.audioHandler.sampleRate,1, 1024))
					self.LFO2.setWavetable(sineWave(self.audioHandler.sampleRate,1, 1024))

					
				if event.key == pygame.K_q:
					self.LFO1.setFrequency(self.LFO1.frequency - 0.1)
				if event.key == pygame.K_w:
					self.LFO1.setFrequency(self.LFO1.frequency + 0.1)
				if event.key == pygame.K_a:
					self.LFO2.setFrequency(self.LFO2.frequency - 0.1)
				if event.key == pygame.K_s:
					self.LFO2.setFrequency(self.LFO2.frequency + 0.1)
				if event.key == pygame.K_e:
					self.LFO1.setPhase(self.LFO1.phase -1)
				if event.key == pygame.K_r:
					self.LFO1.setPhase(self.LFO1.phase +1)
				if event.key == pygame.K_d:
					self.LFO2.setPhase(self.LFO2.phase -1)
				if event.key == pygame.K_f:
					self.LFO2.setPhase(self.LFO2.phase +1)
				
				self.screen.fill((0, 0, 0))
			
			self.wavetablePainter.input(events)

class lissajousDrawer():
	def __init__(self, width, height, xOffset, yOffset, LFO1, LFO2):
		self.width = width
		self.height = height
		self.xOffset = xOffset
		self.yOffset = yOffset
		
		self.LFO1 = LFO1
		self.LFO2 = LFO2
		
	def render(self, samplesToRender):
		LFO1Samples = self.LFO1.render(samplesToRender)
		LFO2Samples = self.LFO2.render(samplesToRender)
		LFO1Samples += 1
		LFO1Samples /= 2
		LFO2Samples += 1
		LFO2Samples /= 2
		
		widthPoints = numpy.asarray(LFO1Samples)
		widthPoints = (widthPoints + self.xOffset) * self.width
		heightPoints = numpy.asarray(LFO2Samples)
		heightPoints = (heightPoints + self.yOffset) * self.height
		
		lissajousPoints = numpy.asarray([[0.0, 0.0]] * len(LFO1Samples), dtype = numpy.int32)
		for i in range(len(lissajousPoints)):
			lissajousPoints[i][0] = widthPoints[i]
			lissajousPoints[i][1] = heightPoints[i]
		
		return [lissajousPoints, widthPoints, heightPoints]


class wavetablePainter():
	
	def __init__(self, width, height, xOffset, yOffset):
		self.width = width
		self.height = height
		self.xOffset = xOffset
		self.yOffset = yOffset
		
		self.clickPosition = (0, 0)
		self.releasePosition = (0, 0)
		self.currentPosition = (0, 0)
		self.previousPosition = (0, 0)
		self.holdingState = False
		
		self.pixels = numpy.full((self.width, 1), self.height)
		
		self.waveTableSize = self.width
		self.waveTable = numpy.zeros(self.waveTableSize)
	
	def input(self, events):
		for event in events:
			if event.type == pygame.MOUSEBUTTONDOWN:
				self.clickPosition = pygame.mouse.get_pos()
				self.holdingState = True
			elif event.type == pygame.MOUSEBUTTONUP:
				self.releasePosition = pygame.mouse.get_pos()
				self.holdingState = False
			
			if event.type == pygame.MOUSEMOTION:
				self.previousPosition = self.currentPosition
				self.currentPosition = pygame.mouse.get_pos()
				
				if self.holdingState:
					if self.currentPosition[0] >= self.xOffset and self.currentPosition[0] < (self.xOffset + self.width):
						if self.currentPosition[1] >= self.yOffset and self.currentPosition[1] < (self.yOffset + self.height):
							if self.previousPosition[0] >= self.xOffset and self.previousPosition[0] < (self.xOffset + self.width):
								if self.previousPosition[1] >= self.yOffset and self.previousPosition[1] < (self.yOffset + self.height):
									self.updatePixelTable()
									self.updateWaveTable()
	
	def updatePixelTable(self):
		xPixelCurrent = round(self.currentPosition[0])
		yPixelCurrent = round(self.currentPosition[1])
		xPixelPrevious = round(self.previousPosition[0])
		yPixelPrevious = round(self.previousPosition[1])
		
		xStart = min(xPixelCurrent, xPixelPrevious)
		xEnd = max(xPixelCurrent, xPixelPrevious)
		
		if xStart == xPixelCurrent:
			yStart = yPixelCurrent
			yEnd = yPixelPrevious
		else:
			yStart = yPixelPrevious
			yEnd = yPixelCurrent
		
		for i in range(xStart, xEnd + 1, 1):
			yValue = round(numpy.interp(i, [xStart, xEnd], [yStart, yEnd]))
			self.pixels[i] = yValue
		
	def render(self):
		return self.pixels
	
	def updateWaveTable(self):
		stepSize = self.width / self.waveTableSize
		increment = 0.0
		
		for i in range(self.waveTableSize):
			frac, whole = math.modf(increment)
			whole = int(whole)
			
			if i != self.waveTableSize - 1:
				yValue = linearInterpolation(self.pixels[whole], self.pixels[whole + 1], frac)
			else:
				yValue = self.pixels[whole]
			
			if yValue != 0:
				yValue = ((yValue - self.yOffset) / (self.height / 2)) - 1
				yValue = -yValue
			else:
				yValue = 1
			
			self.waveTable[i] = yValue
			increment += stepSize
		
		return self.waveTable
	
	def smoothPixelTable(self):
		self.pixels = (self.pixels / (self.height / 2))
		self.pixels = self.pixels - 1
		
		for i in range(16):
			for j in range(2, len(self.pixels) - 2, 1):
				p1 = self.pixels[j - 2]
				p2 = self.pixels[j - 1]
				p3 = self.pixels[j + 1]
				p4 = self.pixels[j + 2]
				
				sum = p1 + p2 + p3 + p4
				sum /= 4
				self.pixels[j] = (self.pixels[j] + sum) / 2
		
		self.pixels = self.pixels + 1
		self.pixels = self.pixels * (self.height / 2)
	
if __name__ == '__main__':
	a = audioHandler()
	g = graphicHandler(a)
	g.start()
