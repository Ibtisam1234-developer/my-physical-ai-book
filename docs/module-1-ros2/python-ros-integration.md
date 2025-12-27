# Python-ROS Integration

## Using rclpy for Python-based ROS Nodes

rclpy is the Python client library for ROS 2. It provides the interface between Python programs and the ROS 2 system, allowing Python developers to create ROS nodes, publishers, subscribers, services, and actions.

## Installing rclpy

rclpy is included with ROS 2 installations. If you installed ROS 2 Humble, rclpy is already available:

```bash
# Verify rclpy installation
python3 -c "import rclpy; print('rclpy version:', rclpy.get_rclpy_version())"
```

## Basic Node Structure with rclpy

```python
#!/usr/bin/env python3
"""
Basic structure of a ROS 2 node using rclpy.
"""

import rclpy
from rclpy.node import Node


class BasicNode(Node):
    """
    A basic ROS 2 node demonstrating the fundamental structure.
    """

    def __init__(self):
        # Initialize the node with a name
        super().__init__('basic_node')

        # Log a message
        self.get_logger().info('Basic node initialized')


def main(args=None):
    """
    Main function to run the node.
    """
    # Initialize rclpy
    rclpy.init(args=args)

    # Create the node
    node = BasicNode()

    try:
        # Keep the node running
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Publishers and Subscribers with rclpy

```python
#!/usr/bin/env python3
"""
Example of publisher and subscriber using rclpy.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TalkerNode(Node):
    """
    A node that publishes messages.
    """

    def __init__(self):
        super().__init__('talker')

        # Create a publisher
        self.publisher_ = self.create_publisher(
            String,
            'chatter',
            10  # QoS history depth
        )

        # Create a timer to publish periodically
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Counter for messages
        self.i = 0

    def timer_callback(self):
        """Callback function that publishes messages."""
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


class ListenerNode(Node):
    """
    A node that subscribes to messages.
    """

    def __init__(self):
        super().__init__('listener')

        # Create a subscription
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10  # QoS history depth
        )

        # Ensure the subscription is active
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        """Callback function for handling received messages."""
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    """Main function that runs both nodes."""
    rclpy.init(args=args)

    # Create both nodes
    talker = TalkerNode()
    listener = ListenerNode()

    try:
        # Run both nodes
        rclpy.spin(talker)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        talker.destroy_node()
        listener.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Services with rclpy

```python
#!/usr/bin/env python3
"""
Example of services using rclpy.
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool


class ServiceServer(Node):
    """
    A node that provides a service.
    """

    def __init__(self):
        super().__init__('service_server')

        # Create a service server
        self.srv = self.create_service(
            SetBool,
            'set_bool_service',
            self.service_callback
        )

    def service_callback(self, request, response):
        """Callback function for handling service requests."""
        if request.data:
            self.get_logger().info('Setting value to True')
            response.success = True
            response.message = 'Value set to True'
        else:
            self.get_logger().info('Setting value to False')
            response.success = True
            response.message = 'Value set to False'

        return response


class ServiceClient(Node):
    """
    A node that calls a service.
    """

    def __init__(self):
        super().__init__('service_client')

        # Create a client for the service
        self.cli = self.create_client(SetBool, 'set_bool_service')

        # Wait for the service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        # Create a timer to call the service periodically
        self.timer = self.create_timer(2.0, self.call_service)
        self.request_counter = 0

    def call_service(self):
        """Function to call the service."""
        # Create a request
        request = SetBool.Request()
        request.data = (self.request_counter % 2) == 0  # Alternate True/False

        # Call the service asynchronously
        self.future = self.cli.call_async(request)
        self.future.add_done_callback(self.service_response_callback)

        self.request_counter += 1

    def service_response_callback(self, future):
        """Callback function for handling service responses."""
        try:
            response = future.result()
            self.get_logger().info(f'Service response: success={response.success}, message={response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')


def main_server(args=None):
    """Main function for the service server."""
    rclpy.init(args=args)
    node = ServiceServer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main_client(args=None):
    """Main function for the service client."""
    rclpy.init(args=args)
    node = ServiceClient()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

## Actions with rclpy

```python
#!/usr/bin/env python3
"""
Example of actions using rclpy.
"""

import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import time


class FibonacciActionServer(Node):
    """
    An action server that computes Fibonacci sequences.
    """

    def __init__(self):
        super().__init__('fibonacci_action_server')

        # Create an action server
        self._action_server = ActionServer(
            self,
            Fibonacci,  # This would be from action definition
            'fibonacci',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )

    def goal_callback(self, goal_request):
        """Accept or reject goals."""
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject cancel requests."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the goal."""
        self.get_logger().info('Executing goal...')

        # Feedback and result messages
        feedback_msg = Fibonacci.Feedback()
        result_msg = Fibonacci.Result()

        # Fibonacci sequence
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result_msg.sequence = feedback_msg.sequence
                self.get_logger().info('Goal canceled')
                return result_msg

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1]
            )

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)  # Simulate work

        # Goal succeeded
        goal_handle.succeed()
        result_msg.sequence = feedback_msg.sequence
        self.get_logger().info('Goal succeeded')

        return result_msg


def main(args=None):
    """Main function for the action server."""
    rclpy.init(args=args)
    node = FibonacciActionServer()

    try:
        # Use a multi-threaded executor to handle multiple goals
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

## Working with Parameters

```python
#!/usr/bin/env python3
"""
Example of working with parameters using rclpy.
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter


class ParameterNode(Node):
    """
    A node that demonstrates parameter usage.
    """

    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('debug_mode', False)
        self.declare_parameter('joint_limits', [1.57, 1.57, 3.14])

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.debug_mode = self.get_parameter('debug_mode').value
        self.joint_limits = self.get_parameter('joint_limits').value

        self.get_logger().info(f'Robot name: {self.robot_name}')
        self.get_logger().info(f'Max velocity: {self.max_velocity}')
        self.get_logger().info(f'Debug mode: {self.debug_mode}')
        self.get_logger().info(f'Joint limits: {self.joint_limits}')

        # Add parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        """Callback for parameter changes."""
        for param in params:
            if param.name == 'max_velocity' and param.type_ == Parameter.Type.DOUBLE:
                if param.value > 5.0:
                    self.get_logger().warn('Max velocity is very high!')

        return SetParametersResult(successful=True)


def main(args=None):
    """Main function for the parameter node."""
    rclpy.init(args=args)
    node = ParameterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Best Practices for Python-ROS Integration

### Error Handling

```python
#!/usr/bin/env python3
"""
Best practices for error handling in rclpy.
"""

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException


class RobustNode(Node):
    """
    A node demonstrating robust error handling.
    """

    def __init__(self):
        super().__init__('robust_node')

        try:
            # Try to get a parameter that might not exist
            self.declare_parameter('critical_param', 1.0)
            self.critical_param = self.get_parameter('critical_param').value
        except ParameterNotDeclaredException:
            self.get_logger().error('Critical parameter not declared')
            raise

        # Use try-catch for operations that might fail
        try:
            self.setup_resources()
        except Exception as e:
            self.get_logger().fatal(f'Failed to setup resources: {e}')
            raise

    def setup_resources(self):
        """Setup resources with error handling."""
        # Resource setup code here
        pass

    def destroy_node(self):
        """Cleanup resources when node is destroyed."""
        try:
            self.cleanup_resources()
        except Exception as e:
            self.get_logger().error(f'Error during cleanup: {e}')
        finally:
            super().destroy_node()

    def cleanup_resources(self):
        """Cleanup resources."""
        # Cleanup code here
        pass
```

## Exercise: Create a Complete Robot Control Node

Create a Python node that:
1. Declares and uses parameters for robot configuration
2. Publishes joint states
3. Subscribes to velocity commands
4. Provides a service for robot status
5. Uses proper error handling and logging

## Learning Outcomes

After completing this section, students will be able to:
1. Create ROS 2 nodes using rclpy
2. Implement publishers, subscribers, services, and actions
3. Work with parameters in ROS 2 nodes
4. Apply best practices for error handling
5. Structure Python code for ROS 2 applications

## Next Steps

Continue to [URDF for Humanoid Robots](./urdf-humanoid-robots.md) to learn about robot modeling with Unified Robot Description Format.