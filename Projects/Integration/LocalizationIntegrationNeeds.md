From IMU Team:
Callibrated IMU Data in an array or a parwut file that shows the (x.y,z) of the accelerometer, gyroscope, and magnetometer

From YOLACT Team:
Array
Each pixel will be each unique cone (Unique identification number, and pass that to Map Localization)
Pixel % 4 = Cone Type
Get you 0,1,2,3
Pixel // 4 = Cone #

From Depth Estimation:
2D Array of Floats
Each pixel of distance in meters

To Path Planning:
2D Array of Cone Location
