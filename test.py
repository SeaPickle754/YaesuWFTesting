import array as arr

import ft4222
from ft4222 import SysClock
from ft4222.SPI import Cpha, Cpol
from ft4222.SPIMaster import Mode, Clock, SlaveSelect
from ft4222.GPIO import Port, Dir
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d, pchip

wf1Offset = 0
wf1Length = 852
wf2Offset = 852
wf2Length = 852
af1fftOffset = 1704
af1fftLength = 192
af1oscilloOffset = 1896
af1oscilloLength = 400
dataChunkOffset = 2888
dataChunkLength = 144

nbDev = ft4222.createDeviceInfoList()
for i in range(nbDev):
  print(ft4222.getDeviceInfoDetail(i, False))

dev = ft4222.openByDescription('FT4222 A')

dev.spiMaster_Init(Mode.SINGLE, Clock.DIV_16, Cpol.IDLE_HIGH, Cpha.CLK_TRAILING, SlaveSelect.SS0)
dev.setClock(SysClock.CLK_48)

fig,(ax,oscax,wfax) = plt.subplots(3,1)
oscline = Line2D([0],[0])
oscax.add_line(oscline)
oscax.set_ylim(0,256)
oscax.set_xlim(0,400)

im = ax.imshow(np.zeros((500,af1fftLength), dtype=np.uint8), vmin=0, vmax=255, aspect='auto', cmap='turbo', interpolation='nearest')
wfim = wfax.imshow(np.zeros((500,wf1Length), dtype=np.uint8), vmin=60, vmax=255, aspect='auto', cmap='turbo', interpolation='nearest')

oscX = np.linspace(1,400, num=400)

plt.ylabel('Time')

def updateWaterfall(i):
# while True:
  readData = dev.spiMaster_SingleRead(4096, False)
  newline = np.frombuffer(readData, dtype=np.uint8, offset=af1fftOffset, count=af1fftLength)
  oscdata = np.frombuffer(readData, dtype=np.uint8, offset=af1oscilloOffset, count=af1oscilloLength)
  wfdata = np.frombuffer(readData, dtype=np.uint8, offset=wf1Offset, count=wf1Length)
  datachunk = np.frombuffer(readData, dtype=np.uint8, offset=dataChunkOffset, count=dataChunkLength)
  np.set_printoptions(formatter={'int':lambda x: format(x, '02X')}, linewidth=99)
  #   print(datachunk)

  f = interp1d(oscX, oscdata, kind='cubic')
  
  arr = np.roll(im.get_array(), -1, axis=0)
  arr[0] = np.invert(newline)
  
  wfarr = np.roll(wfim.get_array(), -1, axis=0)
  wfarr[0] = np.invert(wfdata)

  oscline.set_data([oscX], [oscdata])

  im.set_data(arr)
  wfim.set_data(wfarr)
  ax.relim()
  ax.autoscale_view()

ani = animation.FuncAnimation(fig, updateWaterfall, interval=50, frames=10000)
plt.show()

