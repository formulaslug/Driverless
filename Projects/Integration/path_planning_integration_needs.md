# From Localization
Need a 2-d np.float32 array of cone coordinates from local coordinate frames with coordinates relative to global map. Will also need color confidences of each cone given as a np.float32 probability between blue, yellow, and orange. We also need car pose with both coordinates and current estimated direction given as an angle in degrees relative to the positive global x-axis, both of which are also np.float32. 

# To Control Planning
We are going to provide 2-d array of waypoint coordinates which will be our planned path through the track. We will also provide the curvature at each point, represented by `curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)`, using a cubic spline. Hopefully, we will be able to give path confidence as well.