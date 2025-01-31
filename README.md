# Waterfall Testing Tools
Testing code for pulling waterfall data from the Yaesu FT710, FTDX10, and FT101 series radios.

The FT710 uses a FT4222 USB->SPI/I2C Bridge in SPI mode to send data
The FTDX10 and FT101 use SPI from the 13-pin din ACC connector on the back (more for this coming later)


## Setup
The FT710 needs the SCU-LAN10 option enabled in the settings menu.
The FT4222 gets set up in SPI Master mode, Clock/16, Clock Idle High, Clock Trailing, SS0O with a 48MHz system clock.
No need to send anything to initiate the data stream, but the SCU-LAN10 appears to send 'INIT' (maybe needed for the DX10 and 101)

## Data Format
Just playing around with the data now, figuring out protocol. 
It seems the first 852 bytes is the waterfall data for receiver 1, the next 852 is waterfall data for receiver 2 (on the 101).
Waterfall data is a single line, uint8.

Next is the AF FFT for receiver 1, 192 bytes uint8.
The next 400 bytes is oscilloscope data for AF receiver 1, 400 bytes, uint8 (128 is the zero point for the graph).
Repeat again for receiver 2.

The next 144 bytes is radio parameters, and this is my current main focus. This has frequencies, bandwidths, meter data, filters, etc.

So far determined:
I think the byte 22 from the end is the S meter reading. It is constantly changing and goes to high values when the RF Gain/Sql knob is all the way down.

