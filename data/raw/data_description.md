# Dataset Description

| Column                | Description                
|:----------------------|:---------------------------
| `ui32_objectID`         | Object ID assigned by radar
| `ui16_ageCount`         | Number of frames the Track is alive after being released                       
| `ui16_predictionCount`  |                     
| `ui16_staticCount`      | Number of frames the Track is static. Counts down.                       
| `classID`               | An enumerated classification of vehicle type.
| `f32_trackQuality`      | The Quality of the Track. Depends on different attributes of the assigned target.
| `f32_positionX_m`       | Distance the Track has to the sensor in X direction [m]
| `f32_positionY_m`       | Distance the Track has to the sensor in Y direction [m]
| `f32_velocityInDir_mps` | Velocity of the Track in the direction of the Track [m/s]
| `f32_directionX`        | Direction of the Track in X direction (normalized)
| `f32_directionY`        | Direction of the Track in Y direction (normalized)
| `f32_distanceToFront_m` | Distance to the front of the Tracked Object [m]
| `f32_distanceToBack_m`  | Distance to the back of the Track Object [m]
| `f32_length_m`          | Length of the Tracked Object [m]
| `f32_width_m`           | Width of the Tracked Object [m]
| `epoch_time`            | Time that the tracked object information was recorded 
| `ip`                    | hashed ip address of the radar sensor
| `object_id`             | Object id assigned by the central server. Unique to `ui32_objectID` + `ip` 