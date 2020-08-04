![Banner](https://github.com/jonathanrjpereira/UWB/blob/master/img/Banner.svg)

The coordinates of a moveable Tag node can be estimated given the known locations of a minimum of three fixed Anchor nodes and the distance between each node using a method such as Trilateration.

But for a system consisting of a large network of anchor nodes, it is physically impossible to manually measure the distance between each node accurately. Hence, in order to make an Indoor Localization System scalable, it is necessary to develop a system wherein the Anchor nodes are self-calibrating.

The repository is divided into two sections:
- Distance Measurement & Positioning implementation schemes using Fixed Anchors.
- Anchor Tag Self-Calibration Methods

The implementation schemes and self-calibration methods will be tested using Decawave DWM1001 beacons.

For a detailed step-by-step guide, check out the [Wiki](https://github.com/jonathanrjpereira/UWB/wiki).

![DWM1001](https://github.com/jonathanrjpereira/UWB/blob/master/img/IMG_2021.jpg)

# Table of Contents
## [Ultra-wideband (UWB)](https://github.com/jonathanrjpereira/UWB/wiki/Ultra-wideband-(UWB))
## Implementation Schemes
* Distance Measuring
  * [Time of Flight (ToF)](https://github.com/jonathanrjpereira/UWB/wiki/Time-of-Flight-(ToF))
  * [Two Way Ranging (TWR) - DWM1001](https://github.com/jonathanrjpereira/UWB/wiki/Two-Way-Ranging-(TWR):-Decawave-DWM1001)
* Positioning
  * [Trilateration](https://github.com/jonathanrjpereira/UWB/wiki/Trilateration)

## Self-Calibrating Anchors
* [Multidimensional Scaling (MDS)](https://github.com/jonathanrjpereira/UWB/wiki/Multidimensional-Scaling-(MDS))
* [MDS with 2 DWM1001 Modules](https://github.com/jonathanrjpereira/UWB/wiki/MDS-with-2-DWM1001-Modules)

