import math
import struct

import scipy
import scipy.io.wavfile          # reading in data from image
import scipy.ndimage             # rescaling of image using ndimage.zoom()
import numpy as np               # for the common np.arrays
import matplotlib.pyplot as plt  # For useful plotting of spectogram and periodogram

import wave                      # wave - allows for the reading/writing of audio files with the .wav format

from PIL import Image            # Image Module from Python Imaging Library - allows for the simple manipulation
                                 #      of images. Required for:
                                 #           - opening the image (Image.open("<filename>") )
                                 #           - converting the image to grayscale (Image.open().convert("L"))
import IPython.display           # for playing back audio using IPython.display.Audio()


# Spectogram - freq vs time
def plotSpectogram(file="pic2sound.wav"):
    sample_rate, X = scipy.io.wavfile.read(file)  # returns the sample rate (samp/sec) and the data from the WAV file
    plt.specgram(X, Fs=sample_rate,
                 xextent=(0, 60))  # returns spectrum(2D), freqs(1D), t(1D midpt segment, AKA columns
    print("File: ", file)  # of spectrum), and the Image produced (plotted spectrum).
    print("Sample rate (Hz): ", sample_rate)  # Already computes NFFT for split segments (of array) and returns
    #    a colormap


# Periodogram - power vs freq
def plotPeriodogram(file="pic2sound.wav"):
    sample_rate, X = scipy.io.wavfile.read(file)
    plt.psd(X, Fs=sample_rate)  # returns power spectrum(real mean of segments), freqs, line graph
    print("File: ", file)  # Also, computes FFT
    print("Sample rate (Hz): ", sample_rate)


def plotMat(mat):
    fmat = np.flipud(mat)  # vertically flips a given array
    X, Y = np.meshgrid(range(fmat.shape[0]), range(fmat.shape[1]))  # meshgrid(a, b) => inputs n coordinate arrays
    #               => outputs n dim array of repeated arrays
    # range(1, 5) => sequence of numbers 1 to 5 in order
    Z = fmat[X, Y]  # .shape => (#columns, #rows, etc.)
    plt.pcolormesh(Y, X, Z)
    plt.show()


def loadPicture(size, file, verbose=1):
    img = Image.open(file)
    img = img.convert("L")
    # img = img.resize(size) # DO NOT DO THAT OR THE PC WILL CRASH

    imgArr = np.array(img)
    if verbose:
        print("Image original size: ", imgArr.shape)

    # Increase the contrast of the image
    imgArr = imgArr / np.max(imgArr)
    imgArr = 1 / (imgArr + 10 ** 15.2)

    # Scale between 0 and 1
    imgArr -= np.min(imgArr)
    imgArr = imgArr / np.max(imgArr)

    # Remove low pixel values
    removeLowValues = np.vectorize(lambda x: x if x > 0.02 else 0, otypes=[np.float])
    imgArr = removeLowValues(imgArr)

    if size[0] == 0:
        size = imgArr.shape[0], size[1]
    if size[1] == 0:
        size = size[0], imgArr.shape[1]
    resamplingFactor = size[0] / imgArr.shape[0], size[1] / imgArr.shape[1]
    if resamplingFactor[0] == 0:
        resamplingFactor = 1, resamplingFactor[1]
    if resamplingFactor[1] == 0:
        resamplingFactor = resamplingFactor[0], 1

    # Order : 0=nearestNeighbour, 1:bilinear, 2:cubic etc...
    imgArr = scipy.ndimage.zoom(imgArr, resamplingFactor, order=0)

    if verbose:
        print("Resampling factor", resamplingFactor)
        print("Image resized :", imgArr.shape)
        print("Max intensity: ", np.max(imgArr))
        print("Min intensity: ", np.min(imgArr))
        plotMat(imgArr)
    return imgArr

def lptest(size, file, verbose=1):
    img = Image.open(file).convert("L")

    imgArr = np.array(img)
    if verbose:
        print("Image original size: ", imgArr.shape)

    # Increase the contrast of the image
    imgArr = imgArr + 10 ** 15.2
    imgArr -= np.min(imgArr)
    imgArr = imgArr / np.max(imgArr)

    # Remove low pixel values
    removeLowValues = np.vectorize(lambda x: x if x > 0.02 else 0, otypes=[np.float])
    imgArr = removeLowValues(imgArr)

    if size[0] == 0:
        size = imgArr.shape[0], size[1]
    if size[1] == 0:
        size = size[0], imgArr.shape[1]
    resamplingFactor = size[0] / imgArr.shape[0], size[1] / imgArr.shape[1]
    if resamplingFactor[0] == 0:
        resamplingFactor = 1, resamplingFactor[1]
    if resamplingFactor[1] == 0:
        resamplingFactor = resamplingFactor[0], 1

    # Order : 0=nearestNeighbour, 1:bilinear, 2:cubic etc...
    imgArr = scipy.ndimage.zoom(imgArr, resamplingFactor, order=0)

    if verbose:
        print("Resampling factor", resamplingFactor)
        print("Image resized :", imgArr.shape)
        print("Max intensity: ", np.max(imgArr))
        print("Min intensity: ", np.min(imgArr))
        plotMat(imgArr)
    return imgArr


def genSoundFromImage(file, output="sound.wav", duration=5.0, sampleRate=44100.0):
    wavef = wave.open(output, 'w')
    wavef.setnchannels(1)  # mono
    wavef.setsampwidth(2)
    wavef.setframerate(sampleRate)

    max_frame = int(duration * sampleRate)
    max_freq = 22000  # Hz
    max_intensity = 32767

    stepSize = 400  # Hz
    steppingSpectrum = int(max_freq / stepSize)

    imgMat = loadPicture((steppingSpectrum, max_frame), file, verbose=0)
    imgMat *= max_intensity
    print("Input: ", file)
    print("Duration (in seconds): ", duration)
    print("Sample rate: ", sampleRate)
    print("Computing each soundframe sum value..")
    for frame in range(max_frame):
        if frame % 60 == 0:  # Only print once in a while
            IPython.display.clear_output(wait=True)
            print("Progress: ==> {:.2%}".format(frame / max_frame), end="\r")
        signalValue, count = 0, 0
        for step in range(steppingSpectrum):
            intensity = imgMat[step, frame]
            if intensity == 0:
                continue
            # nextFreq is less than currentFreq
            currentFreq = max_freq - step * stepSize
            nextFreq = max_freq - (step + 1) * stepSize
            if nextFreq < 0:  # If we're at the end of the spectrum
                nextFreq = 0
            for freq in range(nextFreq, currentFreq, 1000):  # substep of 1000 Hz is good
                signalValue += intensity * math.cos(freq * 2 * math.pi * float(frame) / float(sampleRate))
                count += 1
        if count == 0: count = 1
        signalValue /= count

        data = struct.pack('<h', int(signalValue))
        wavef.writeframesraw(data)

    wavef.writeframes(''.encode())
    wavef.close()
    print("\nProgress: ==> 100%")
    print("Output: ", output)

def gsfitest(file, output="sound.wav", duration=5.0, sampleRate=44100.0):
    wavef = wave.open(output, 'w')
    wavef.setnchannels(1)  # mono
    wavef.setsampwidth(2)
    wavef.setframerate(sampleRate)

    max_frame = int(duration * sampleRate)
    max_freq = 22000  # Hz
    max_intensity = 32767  # max byte

    stepSize = 400  # Hz
    steppingSpectrum = int(max_freq / stepSize)

    imgMat = lptest((steppingSpectrum, max_frame), file, verbose=0)
    imgMat *= max_intensity
    print("Input: ", file)
    print("Duration (in seconds): ", duration)
    print("Sample rate: ", sampleRate)
    print("Computing each soundframe sum value..")
    for frame in range(max_frame):
        if frame % 60 == 0:  # Only print once in a while
            IPython.display.clear_output(wait=True)
            print("Progress: ==> {:.2%}".format(frame / max_frame), end="\r")
        signalValue, count = 0, 0
        for step in range(steppingSpectrum):
            intensity = imgMat[step, frame]
            if intensity == 0:
                continue
            # nextFreq is less than currentFreq
            currentFreq = max_freq - step * stepSize
            nextFreq = max_freq - (step + 1) * stepSize
            if nextFreq < 0:  # If we're at the end of the spectrum
                nextFreq = 0
            for freq in range(nextFreq, currentFreq, 1000):  # substep of 1000 Hz is good
                signalValue += intensity * math.cos(freq * 2 * math.pi * float(frame) / float(sampleRate))
                count += 1
        if count == 0: count = 1
        signalValue /= count

        data = struct.pack('<h', int(signalValue))
        wavef.writeframesraw(data)

    wavef.writeframes(''.encode())
    wavef.close()
    print("\nProgress: ==> 100%")
    print("Output: ", output)







#User interface
isImage = 'n'
userPath = 'none'
while (isImage == 'n' or 'N'):
    userPath = str(input("Enter the image file path you wish to convert: ") )
    IPython.display.Image(userPath)
    isImage = chr(input('\n Is this the image you wish to convert? (Y)es or (N)o: ') )

print('Good')

"""
lptest(size=(2901,2300), file="/Users/imman/Downloads/Rainbow.png")
gsfitest(file="/Users/imman/Downloads/Rainbow.png")
IPython.display.Audio("sound.wav")

"""