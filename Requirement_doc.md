
ThoughtLink - From Brain to Robot
VC Track supported by Kernel & Dimensional

1. Goals and Motivation
As AI systems improve, the hardest problems are shifting from perception and generation to real-time execution and coordination. In robotics, this gap is especially pronounced. Robots can already see, plan, and act—but when ambiguity arises, humans are still forced to intervene through slow, low-bandwidth interfaces.
Imagine a factory with 1000 humanoid robots operating simultaneously. Each robot performs tasks like assembly, inspection, navigation, or logistics. Most of the time, autonomy works. But when something unexpected happens—an obstacle, a misplaced object, a failed task—the system needs human judgment, immediately.
Today, that judgment flows through keyboards, dashboards, and manual overrides. This does not scale.
Some approaches, such as Neuralink, aim to solve this with invasive neural implants. ThoughtLink explores a complementary and immediately deployable path: decoding high-level human intent from non-invasive brain signals and mapping it directly to executable robot instructions.
The goal is not to control motors or joints. It is to create an intent and orchestration layer that allows a single human to reason once and intervene across dozens—or eventually hundreds—of robots at machine speed.
This challenge is an invitation to explore that layer and define what fast, reliable, and scalable brain-to-robot control should look like.

2. Hero Features — What You Could Build
ThoughtLink enables capabilities that are not possible with traditional interfaces.
Core Capability: Brain-to-Instruction Mapping
Decode non-invasive brain signals into discrete, high-level robot instructions
Treat neural activity as a time-series intent signal, not raw control
Map human cognition to actions robots already know how to execute
Orchestration-Level Outcomes
Intent-level control instead of teleoperation
One-to-many supervision, where one human oversees many robots
Ultra-low-latency intervention when autonomy encounters ambiguity
Human judgment at machine scale, without micromanagement
Your challenge: build a system that demonstrates fast, stable, and accurate decoding of human intent, suitable for controlling robots in real time.

3. Example Directions
Pick one direction and go deep.
Temporal Intent Decoding
Decode left, right, forward, and backward intent from brain-signal time series, using smoothing or hysteresis to prevent oscillation. Validate behavior in humanoid simulation.
Hierarchical Intent Models
First detect “movement intent” vs rest, then classify direction to reduce false triggers.
Phase-Aware Modeling
Model intent as phases (initiation, sustain, release), capturing transitions rather than single-frame predictions.
Interpretability & Visualization
Learn latent embeddings of brain activity and visualize how intent evolves over time.
Success is defined by clarity, stability, and timing of intent, not robot motion complexity.

4. How to Think About This
Think of AI as an intent interpreter, not a controller.
In this paradigm:
Robots already know how to move
Humans decide what should happen
AI decides when intent is real, what it maps to, and whether it should be executed
The brain signal is a multivariate time series. The central challenge is balancing accuracy, latency, and stability under real-time constraints.
We are especially interested in how you reason about:
Binary vs multi-class decoding
Temporal context vs instantaneous prediction
Confidence thresholds and false-trigger suppression
When complexity helps—and when it hurts inference speed

5. Getting Started & Resources (Start Simple, Then Scale)
You are strongly encouraged to start simple and only add complexity when it measurably improves performance.
Suggested Progression
Begin with binary classification (e.g., left vs right, move vs rest)
Use short, fixed-length temporal windows
Try classical supervised learning first
Strong baselines may come from:
Logistic regression
Linear or kernel SVMs
Random forests
Simple CNNs over time–channel representations
Once a baseline works, you may:
Expand to multi-class instruction decoding
Add explicit temporal models (RNNs, state-space models)
Model longer-range dependencies using transformers, if justified by the data
Transformers are a valid option—but fast inference matters more than model size.
Provided Infrastructure
Pre-recorded brain dataset (~5 hours)
Curated non-invasive brain recordings covering multiple actions and intent phases.
Humanoid robot simulation
A simulation environment that accepts discrete, high-level instructions.
Execution & testing script
A ready-made script that connects your model’s predictions directly to the humanoid simulation for closed-loop testing.
Optional Reference Material
https://github.com/Nabla7/brain-robot-interface
Example end-to-end brain–robot interface pipelines.
https://huggingface.co/datasets/KernelCo/robot_control/blob/main/README.md
Dataset, Documentation of robot command spaces and simulation execution.
Guiding principle:
A fast, simple model that works beats a slow, complex model that doesn’t.

6. Evaluation Criteria
Projects will be evaluated primarily on model performance under real-time constraints.
Intent Decoding Performance
Accuracy and robustness of brain-to-instruction mapping.
Inference Speed & Latency
End-to-end prediction latency. Can the model support real-time control?
Temporal Stability
Smooth handling of transitions without oscillation or flicker.
False Trigger Rate & Confidence Handling
Explicit use of thresholds, debouncing, or hysteresis to prevent unintended actions.
Scalability of the Approach
Could this model realistically support a system supervising 100 humanoid robots?
Demo Clarity
Is the intent-to-action loop clearly demonstrated in simulation?
Bonus consideration for teams that quantify latency–accuracy tradeoffs, compare simple vs complex models, or surface failure modes and open research questions.

7. Why It Matters
In a world with 1000 humanoid robots operating in a single facility, autonomy alone is not enough. What matters is how quickly and safely human judgment can be injected when something goes wrong.
ThoughtLink explores the missing intelligence layer that makes this possible: a fast, non-invasive channel from human intent to robotic action.
If successful, one human can supervise dozens—or hundreds—of robots, intervening only where context and judgment are required.
This is not teleoperation.
Not automation.
It is intent infrastructure for intelligent machines—
from brain to robot, at the speed of thought.

