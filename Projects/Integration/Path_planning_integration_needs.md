# From Localization
Need a 2-d array of cone coordinates from local coordinate frames with coordinates relative to global map. Will also need color confidences of each cone as well as car pose with both coordinates and current estimated direction given as an angle relative to the global x-axis. Coordinate data types are flexible.

# To Control Planning
Going to provide 2-d array of waypoints which will be our planned path through the track. We will also provide the curvature at each point using a cubic spline. Hopefully, we will be able to give target velocies at each waypoint as well, but I'm not sure how useful that would be. 
