import math
import struct

import scipy
import scipy.io.wavfile          # reading in data from image
import scipy.ndimage             # rescaling of image using ndimage.zoom()
import numpy as np               # for the common np.arrays
import matplotlib.pyplot as plt  # For useful plotting of spectogram and periodogram
from matplotlib.image import imread
import os
import wave                      # wave - allows for the reading/writing of audio files with the .wav format

from PIL import Image            # Image Module from Python Imaging Library - allows for the simple manipulation
                                 #      of images. Required for:
                                 #           - opening the image (Image.open("<filename>") )
                                 #           - converting the image to grayscale (Image.open().convert("L"))
import IPython.display           # for playing back audio using IPython.display.Audio()


# Spectogram - freq vs time
def plotSpectogram(file="Img2Sound.wav"):
    sample_rate, X = scipy.io.wavfile.read(file)  # returns the sample rate (samp/sec) and the data from the WAV file
    spec, freq, t, im = plt.specgram(X, Fs=sample_rate,
                 xextent=(0, 60))  # returns spectrum(2D), freqs(1D), t(1D midpt segment, AKA columns
    plt.savefig("SoundSpecgram.png")
    plt.close()
    print("File: ", file)  # of spectrum), and the Image produced (plotted spectrum).
    print("Sample rate (Hz): ", sample_rate)  # Already computes NFFT for split segments (of array) and returns
    return freq, spec
    #    a colormap


# Periodogram - power vs freq
def plotPeriodogram(file="Img2Sound.wav"):
    sample_rate, X = scipy.io.wavfile.read(file)
    y, x = plt.psd(X, Fs=sample_rate)  # returns power spectrum(real mean of segments), freqs, line graph
    plt.savefig("Actualperiodogram.png")
    plt.close()
    # print(np.min(x))
    print("File: ", file)  # Also, computes FFT
    print("Sample rate (Hz): ", sample_rate)
    return x, y


def plotMat(mat, name):
    fmat = np.flipud(mat)  # vertically flips a given array
    X, Y = np.meshgrid(range(fmat.shape[0]), range(fmat.shape[1]))  # meshgrid(a, b) => inputs n coordinate arrays
    #               => outputs n dim array of repeated arrays
    # range(1, 5) => sequence of numbers 1 to 5 in order
    Z = fmat[X, Y]  # .shape => (#columns, #rows, etc.)
    fig = plt.figure()
    plt.pcolormesh(Y, X, Z)
    plt.colorbar()
    plt.title(name)
    plt.axis("off")
    fig.savefig(name + '.png')
    plt.close()
    #plt.show()


def loadPicture(attenuation, size, file, verbose=1):
    img = Image.open(file).convert("L")
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
    removeLowValues = np.vectorize(lambda x: x if x > attenuation else 0, otypes=[np.float])
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
        plotMat(imgArr, "Attenuated Image")
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


def genSoundFromImage(duration, file, output="Img2Sound.wav",  sampleRate=44100.0):
    wavef = wave.open(output, 'w')
    wavef.setnchannels(1)  # mono
    wavef.setsampwidth(2)
    wavef.setframerate(sampleRate)

    max_frame = int(duration * sampleRate)
    max_freq = 22000  # Hz
    max_intensity = 32767

    stepSize = 400  # Hz
    steppingSpectrum = int(max_freq / stepSize)

    imgMat = loadPicture(atten, (steppingSpectrum, max_frame), file, verbose=0)
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
dur = 5
atten = .3
num = 0
isImage = 'n'
userPath = 'PATH unfound'
while (isImage == 'n' or isImage =='N' ):
    userPath = str(input("Enter the image file path you wish to convert: ") )
    testIm = imread(userPath)
    print("Exit window to proceed \n")
    plt.imshow(testIm) #**data[dtype])
    plt.axis('off')
    plt.show()
    isImage = str(input('Is this the image you wish to convert? (Y)es or (N)o: '))

sec = input("Input the duration of the audio in seconds (Default is 5 sec): \n WARNING: A longer duration will cause "
            "the run-time to be much longer; a duration longer than the default is not recommended. ")

while num == 0 or atten > 1:
    num = 1
    atten =  input("Input the signal attenuation on a scale of 0 to 1 (Default is .3): ")
    if atten == '':
        atten = .3
        break
    else:
        atten = abs(float(atten))

if sec == '':
    dur = 5
else:
    dur = float(sec)



name = os.path.dirname(userPath) #os.getcwd()
newdir = os.path.join(name, "Image2Sound")

try:
    os.makedirs(newdir)
except OSError:
    print("Unable to create folder.")

os.chdir(newdir)
imgMat = loadPicture(atten, size=(2901,2300), file=userPath)
genSoundFromImage(dur, file=userPath)
f, intensity, = plotSpectogram(file="Img2Sound.wav")
w = f
w = w / np.max(w)  # **14)
w = w * (3.89)
w = w + 4  # **14)
plt.plot(w,intensity)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.title("Color Intensity")
plt.legend()
plt.savefig("Light Intensity.png")
plt.close()

plotPeriodogram(file="Img2Sound.wav")
print() #os.rename(newdir, "ImageData(Immanuel)")



"""
lptest(size=(2901,2300), file="/Users/imman/Downloads/Rainbow.png")
gsfitest(file="/Users/imman/Downloads/Rainbow.png")
IPython.display.Audio("sound.wav")


"""