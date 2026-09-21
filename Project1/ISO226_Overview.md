# ISO 226 and Equal Loudness Curves

**Wenrui Chen, EN.520.645 Audio Signal Processing, Project 1**

### What is ISO 226?

ISO 226 is an international standard that describes how loud pure tones sound to a person with normal hearing. It was made because a sound level meter only measures physical pressure, but our ears do not respond to all frequencies in the same way. Two tones with exactly the same sound pressure level can sound very different in loudness if one is at 100 Hz and the other at 1 kHz, so a measured level alone does not tell us how loud something sounds. To have a common reference for this, the standard collected listening test results from many young listeners with normal hearing, who listened to pure tones with both ears in a free field, and combined them into one set of curves. 

The first curves of this kind were measured by Fletcher and Munson in 1933, which is why they are also called Fletcher-Munson curves; Robinson and Dadson repeated the measurement in 1956, that version became ISO 226 in 1987, and the current 2003 edition replaced it with new data.

The curves are plotted with frequency on the horizontal axis, on a logarithmic scale from 20 Hz to 12.5 kHz, and sound pressure level in dB SPL on the vertical axis. Each single curve is one loudness level, measured in phon. A tone has a loudness level of N phon if it sounds as loud as a 1 kHz tone at N dB SPL, so the 1 kHz tone is the reference and at 1 kHz the phon value equals the dB SPL value. 

To draw one curve, the listener hears the 1 kHz reference tone at a fixed level, then a test tone at another frequency, and the level of the test tone is adjusted until both sound equally loud. Repeating this at every test frequency and connecting the points gives the curve for that loudness level. The lowest curve, 0 phon, is the hearing threshold, and the standard provides curves from 0 up to 90 phon.

### My understanding of equal loudness curves

Equal-loudness curves are these lines of constant loudness. Every point on one curve sounds equally loud, even though the sound pressure level changes a lot along the curve. All of the curves share the same basic shape. They are high on the left, because the ear is not sensitive to low frequencies, and a 20 Hz tone needs far more pressure than a 1 kHz tone to sound the same. They reach their lowest point between about 2 and 5 kHz, which is where the ear is most sensitive, and then rise again at high frequencies. 

Another feature is that the curves are not parallel. At low frequencies they are spread far apart, and at high loudness levels they come closer together, which means the ear's sensitivity depends on the loudness level, not only on the frequency. From an ELC plot we can read several things. For any frequency we can read how many dB SPL a tone needs to reach a given loudness, and by comparing two frequencies on the same curve we can read how much more or less level one of them needs to sound as loud as the other. By looking at the vertical distance between curves at one frequency we can see how quickly loudness grows with level there. 

<img src="./figures/Standard ELC.png" alt="Standard ELC" style="zoom:18%;" />
