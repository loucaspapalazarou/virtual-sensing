# Virtual Sensing through Transformers and Selective State Spaces
***MSc Dissertation | The University of Edinburgh***

[Entire Document](Virtual_Sensing.pdf)

### Abstract

This dissertation investigates the application of modern machine learning techniques,
specifically Transformers and Selective State Spaces (Mamba), to replicate and replace
physical sensors in robotic systems through virtual sensing. The primary objective is to
develop models capable of inferring sensor outputs from a subset of sensors, thereby
reducing the number of physical sensors needed and lowering associated costs. The
study involves setting up a simulation environment using the Franka Emika Panda robot
to perform a defined task, generating data on various measurements including positions,
orientations, force sensor outputs, and images. Multiple experiments were conducted to
evaluate the effectiveness of the chosen architectures, with a focus on understanding the
impact of model complexity and context size on performance. Despite facing challenges
related to model complexity, data handling, and computational resources, the research
provides valuable insights into the feasibility of virtual sensing. The results indicate
that the models could only predict the average sensor values and struggled to capture
detailed sensor data nuances, highlighting the need for more advanced models and better
training techniques. This work lays the foundation for future exploration in reducing
robotic sensor costs through virtual sensing, with recommendations for employing more
complex models and optimizing data handling strategies

### Usage Example

Use the `train.py` file to train a specified model. Use `-h` for the list of available options.

```bash
python train.py --model transformer
```

