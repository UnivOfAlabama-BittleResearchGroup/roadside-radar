# Dataset Description

| Column            | Description                
|:------------------|:---------------------------
| epoch_time        | Time that data was recorded on central server 
| vehicle_id        | Vehicle ID assigned to 1 or more tracklets                              
| lane              | Lane that the vehicle is classified in
| time_diff         | The time difference between the current frame and the previous frame
| s_smooth          | RTS smoothed $s$ position in the Frenet Frame [m]
| s_velocity_smooth | RTS smoothed $s$ velocity in the Frenet Frame [m/s]
| s_accel_smooth    | RTS smoothed $s$ acceleration in the Frenet Frame [m/s/s]
| d_smooth          | RTS smoothed $d$ position in the Frenet Frame [m]
| d_velocity_smooth | RTS smoothed $d$ velocity in the Frenet Frame [m/s]
| d_accel_smooth    | RTS smoothed $d$ acceleration in the Frenet Frame [m/s/s]
| s_lane            | The nearest discretized $s$ point                
| d_cutoff          | The current midpoint in the $d$ dimension between centerlines
| d_other_center    | The distance to the other centerline [m]
| object_id         | A list of the object ids that were associtation at this time
| length_s          | Length of the vehicle projected on the $s$ axis [m]
| distanceToFront_s | Distance to the front of the vehicle projected on the $s$ axis [m]
| distanceToBack_s  | Distance to the back of the vehicle projected on the $s$ axis [m]
| prediction        | Whether or not this state is a prediction or posterior
| front_s_smooth    | RTS smoothed $s$ position of the front of the vehicle in the Frenet Frame [m]
| back_s_smooth     | RTS smoothed $s$ position of the back of the vehicle in the Frenet Frame [m]
| x_lane            | The nearest discretized $x$ point on the lane centerline
| y_lane            | The nearest discretized $y$ point on the lane centerline
| front_x_smooth    | RTS smoothed $x$ position of the centroid of the vehicle in the Frenet Frame [m]
| front_y_smooth    | RTS smoothed $y$ position of the centroid of the vehicle in the Frenet Frame [m]
| back_x_smooth     | RTS smoothed $x$ position of the back of the vehicle in the Frenet Frame [m]
| back_y_smooth     | RTS smoothed $y$ position of the back of the vehicle in the Frenet Frame [m]
| centroid_x_smooth | RTS smoothed $x$ position of the centroid of the vehicle in the Frenet Frame [m]
| centroid_y_smooth | RTS smoothed $y$ position of the centroid of the vehicle in the Frenet Frame [m]