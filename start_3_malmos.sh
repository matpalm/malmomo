#!/usr/bin/env bash
set -ex

cd /data2/malmo_1/latest/Minecraft/
./launchClient.sh -port 11100 &

cd /data2/malmo_2/latest/Minecraft/
./launchClient.sh -port 11200 &

cd /data2/malmo_3/latest/Minecraft/
./launchClient.sh -port 11300 &
