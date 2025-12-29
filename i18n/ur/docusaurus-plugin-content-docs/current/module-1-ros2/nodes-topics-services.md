---
slug: /module-1-ros2/nodes-topics-services
title: "Nodes، Topics اور Services"
hide_table_of_contents: false
---

# Nodes، Topics اور Services

ROS 2 کی بنیادی communication patterns کو سمجھنا robot software development کے لیے essential ہے۔

## Nodes

ROS 2 میں node ایک executable program ہے جو ROS graph میں register ہوتا ہے۔ Nodes modularity provide کرتے ہیں۔

### Node کی مثال (Python)

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.get_logger().info('Node started!')

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

### Node کی مثال (C++)

```cpp
#include <rclcpp/rclcpp.hpp>

class MyNode : public rclcpp::Node {
public:
    MyNode() : Node("my_node") {
        RCLCPP_INFO(this->get_logger(), "Node started!");
    }
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MyNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
```

## Topics

Topics publish-subscribe model use کرتے ہیں۔ یہ continuous data streams کے لیے ideal ہیں۔

### Publisher

```python
from rclpy.publisher import Publisher

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.publisher = self.create_publisher(String, 'topic_name', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello ROS 2!'
        self.publisher.publish(msg)
```

### Subscriber

```python
class SubscriberNode(Node):
    def __init__(self):
        super().__getattr__('subscriber_node')
        self.subscription = self.create_subscription(
            String, 'topic_name', self.listener_callback, 10)

    def listener_callback(self, msg):
        self.get_logger().info(f'Received: {msg.data}')
```

## Services

Services request-response model use کرتے ہیں۔ یہ synchronous operations کے لیے useful ہیں۔

### Service Server

```python
from rclpy.service import Service
from std_srvs.srv import SetBool

class ServiceNode(Node):
    def __init__(self):
        super().__init__('service_node')
        self.srv = self.create_service(SetBool, 'service_name', self.service_callback)

    def service_callback(self, request, response):
        if request.data:
            response.success = True
            response.message = 'Operation successful!'
        return response
```

### Service Client

```python
class ClientNode(Node):
    def __init__(self):
        super().__init__('client_node')
        self.client = self.create_client(SetBool, 'service_name')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available...')

    def send_request(self):
        request = SetBool.Request()
        request.data = True
        future = self.client.call_async(request)
        return future
```

## اگلے steps

[middleware-real-time-control.md](./middleware-real-time-control.md) پڑھیں تاکہ ROS 2 کے middleware اور real-time control کے بارے میں جان سکیں۔