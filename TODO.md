Code
----------
- !!!!!! FIX TRAINING !!!!!!
- Train RNN (baseline)
- Visualize sensors
- Train with less targets (more accuracy?)
- Request cirrus hours?
- Move stride logic to dataset/dataloader, so that one train step is one optim step
- Find a way to not process images so much. (new branch, pass images through resnet without last layer, create new dataset, NEED TO CHANGE DOCUMENT TOO!)
    - No need since realtime images will be coming in, so need ability for real-time image procesing.
- YOU ARE LOADING THE FILE FOR EACH ENV!!!!

Document
----------

- Task is very specific, could be used for robots that do specific tasks, ideally trained with thousands of tasks
- Add figures of signals from sensors 7,8,9 (or whtever target is) and then figures from predicted signals using all my models
- Say that there are a lot of approaches (seq2seq etc) justify why we went with our specific ones
- (Future work) Principal component analysis (PCA) algorithm, add that inthe future work, curse of dimensionality
- Problem formalization in more detail, use letter notations and stuff like that
- Add that inference is slow, say it in drawbacks etc
- Image processing each step, slow
