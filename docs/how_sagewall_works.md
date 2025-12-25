### The Core Concept: Anomaly vs. Signature
Traditional firewalls work like a club bouncer with a "Banned List." If a person is on the list, they are blocked. This is called **Signature Detection**. It works well for known criminals, but it fails completely against a new criminal who has never been seen before.

SageWall is different. It uses **Anomaly Detection**. Instead of memorizing a list of bad IPs, it learns what "Normal Behavior" looks like. If a packet arrives that mathematically deviates from this normal pattern (even if it comes from a trusted IP), SageWall flags it as suspicious.

### The Architecture (Serverless & Decoupled)
SageWall is built on an Event-Driven Architecture on AWS, separating the "Brain" from the "Body."

**1. The Brain (Inference Layer)**
* **Service:** Amazon SageMaker
* **Model:** XGBoost (Extreme Gradient Boosting)
* **Role:** This is the mathematical engine. It hosts the trained model on an ephemeral EC2 instance. It accepts a list of 41 numbers and returns a single probability score (0.0 to 1.0) in under 100ms.

**2. The Body (Ingestion Layer)**
* **Service:** AWS Lambda (simulated via Boto3 in this demo)
* **Role:** It acts as the traffic controller. It takes the raw user input, formats it into a CSV payload, and securely transmits it to the SageMaker endpoint.

### The Data: "DNA" of a Packet
Network traffic is usually just logs of text. Machines cannot understand text, so we convert every packet into a **Feature Vector**—a list of 41 numerical properties that describe the connection.

**Key Features utilized by the Model:**
* **Duration:** How long the connection lasted.
* **Source Bytes:** The volume of data sent.
* **Count:** The number of connections to the same host in the past 2 seconds (useful for detecting DDoS attacks).
* **Protocol:** TCP, UDP, or ICMP (converted to numerical identifiers).

### The Math: XGBoost
We use the **XGBoost Algorithm**, which is a powerful implementation of "Gradient Boosted Decision Trees."
* Imagine 100 small decision trees asking simple questions (e.g., "Is duration > 2 seconds?", "Is byte count > 500?").
* Each tree votes on whether the packet is malicious.
* XGBoost combines these weak votes into a single, highly accurate prediction.
* Our model was trained on the **NSL-KDD Dataset**, a benchmark cybersecurity dataset containing over 125,000 records of real network traffic.