#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_multiply as quatmult
from tf.transformations import quaternion_inverse as quatinv
from tf.transformations import quaternion_conjugate as quatconj

class STLPublisher:
    def __init__(self):
        # Initialize the node
        rospy.init_node('stl_publisher_node', anonymous=True)

        # Define the subscriber to the odometry topic
        self.odom_sub = rospy.Subscriber('/lidar_0/odom', Odometry, self.odom_callback)

        # Define the publisher to RViz Marker topic
        self.marker_pub = rospy.Publisher('/cartinbot_marker', Marker, queue_size=10)

        # Initialize the marker
        self.marker = Marker()
        self.init_marker()

    def init_marker(self):
        # Set the frame of reference (usually "map" or "odom")
        self.marker.header.frame_id = "world"

        # Type of marker (MESH_RESOURCE for STL)
        self.marker.type = Marker.MESH_RESOURCE

        # Marker ID and lifetime (leave 0 for never expire)
        self.marker.id = 0
        self.marker.lifetime = rospy.Duration()

        # Set the mesh resource (path to the STL file)
        self.marker.mesh_resource = "package://gptr/scripts/cartinbot.stl"

        # Scale (adjust depending on your STL file size)
        self.marker.scale.x = 1.0
        self.marker.scale.y = 1.0
        self.marker.scale.z = 1.0

        # Color of the model
        self.marker.color.r = 0.0
        self.marker.color.g = 1.0
        self.marker.color.b = 0.0
        self.marker.color.a = 1.0

    def odom_callback(self, msg):
        
        # Extract the position and orientation from the odometry message
        orientation = msg.pose.pose.orientation
        q_W_L = [orientation.x, orientation.y, orientation.z, orientation.w]
        q_B_L = [0, 0.38268343236, 0, 0.92387953251] # 45-degree pitch
        q_W_B = quatmult(q_W_L, quatinv(q_B_L))
        orientation.x = q_W_B[0]
        orientation.y = q_W_B[1]
        orientation.z = q_W_B[2]
        orientation.w = q_W_B[3]

        position = msg.pose.pose.position
        position.x = position.x
        position.y = position.y
        position.z = position.z

        # Rotate the vehicle
        # Set the marker's pose to the robot's current position and orientation
        self.marker.pose.position = position
        self.marker.pose.orientation = orientation

        # Update the timestamp for RViz
        self.marker.header.stamp = rospy.Time.now()

        # Publish the marker to RViz
        self.marker_pub.publish(self.marker)

    def run(self):
        # Keep the node alive
        rospy.spin()

if __name__ == '__main__':
    try:
        node = STLPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
