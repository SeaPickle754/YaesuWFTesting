import array as arr

import ft4222
from ft4222 import SysClock
from ft4222.SPI import Cpha, Cpol
from ft4222.SPIMaster import Mode, Clock, SlaveSelect
from ft4222.GPIO import Port, Dir

import numpy as np

dataChunkOffset = 2888
dataChunkLength = 144

nbDev = ft4222.createDeviceInfoList()
for i in range(nbDev):
  print(ft4222.getDeviceInfoDetail(i, False))

dev = ft4222.openByDescription('FT4222 A')

dev.spiMaster_Init(Mode.SINGLE, Clock.DIV_16, Cpol.IDLE_HIGH, Cpha.CLK_TRAILING, SlaveSelect.SS0)
dev.setClock(SysClock.CLK_48)
data = bytes(4096)
init = bytes('INIT', 'utf8')


dev.spiMaster_SingleWrite(init, False)

while True:
  firstData = dev.spiMaster_SingleRead(4096, False)
  input('Make a change, press enter')
  secondData = dev.spiMaster_SingleRead(4096, False)

  firstChunk = np.frombuffer(firstData, dtype=np.uint8, offset=dataChunkOffset, count=dataChunkLength)
  secondChunk = np.frombuffer(secondData, dtype=np.uint8, offset=dataChunkOffset, count=dataChunkLength)

  np.set_printoptions(formatter={'int':lambda x: format(x, '02X')}, linewidth=99)
  print('First:')
  print(firstChunk)
  print('')
  print('Second:')
  print(secondChunk)
  print('')
  print('Diff:')
  print(np.diff([firstChunk, secondChunk], axis=0))
  input('Press enter when ready for another')