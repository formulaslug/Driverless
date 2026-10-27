# Step 1 - Run through Ubuntu Setup (oem-config)

First, connect the following to the developer kit;

1. DisplayPort cable attached to a computer monitor (8)
2. For a monitor with HDMI input, use an active DisplayPort to HDMI adapter/cable.
3. USB keyboard and mouse (12)
4. Ethernet cable (6) (optional if you plan to connect to the Internet via WLAN)
5. Then connect the included power supply into the USB Type-C™ port above the DC jack (4)

Your developer kit should automatically power on, and the white LED (0) near the power button will light. If not, press the Power button (1).
Wait up to 1 minute to have Ubuntu screen on the computer monitor.

1. Review and accept NVIDIA Jetson software EULA
2. Select system language, keyboard layout, and time zone
3. Create username, password, and computer name
4. Configure wireless networking


# Step 2 - Install JetPack Components

Once the initial setup is complete, you can install the latest JetPack components that correspond to your L4T version from the Internet.

Open a terminal window if you are on Ubuntu desktop ( Ctrl+Alt+T ).


***1. Check your L4T version first to see if you have a unit flashed with older version of the BSP.***

`cat /etc/nv_tegra_release`


You may get something like this, # R34 (release), REVISION: 1.0, GCID: 30102743, BOARD: t186ref, EABI: aarch64, DATE: Wed Apr 6 19:11:41 UTC 2022, and this shows that you have L4T for JetPack 5.0 Developer Preview.

***2. If you have an earlier version of L4T, issue the following command to manually put the apt repository entries using commands below.***

`sudo bash -c 'echo "deb https://repo.download.nvidia.com/jetson/common r34.1 main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list'`

`sudo bash -c 'echo "deb https://repo.download.nvidia.com/jetson/t234 r34.1 main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list'`

If you see R34 (release), REVISION: 1.0 or newer, then your apt sources lists are already up to date and you can proceed


***3. Issue the following commands to install JetPack components.***

`sudo apt update`

`sudo apt dist-upgrade`

`sudo reboot`

`sudo apt install nvidia-jetpack`

It can take about an hour to complete the installation (depending on the speed of your Internet connection)