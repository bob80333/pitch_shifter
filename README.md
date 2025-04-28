# Pitch Shift Artifact Remover

The datasets used are VCTK, and VocalSet.  They are permissively licensed datasets.

everything with "_remove_artifacts" is about models that remove pitch shifting artifacts.
they are trained by shifting audio up and then down, which creates audio at same pitch as original, but with all the artifacts of original.
the problem is that they may not generalize to fixing audio at pitches they weren't trained at, since human speech only has a certain range of pitch.

The models seem to have some generalization, but to improve it, singing data from vocalset is added, as well as shifting across a broader range (up and down or down and up).

everything with "_unshift" is about models that directly undo a pitch shift (training the model to shift the audio, not cleanup artifacts).
They do not work well, except in the GAN setting, which is much slower to train.  Overall, results for directly shifting are much worse than using an existing algorithm to shift, and the model to cleanup the artifacts as a second stage.
Update: If you train in the GAN setting for ~1 million steps, the results are pretty good, but takes a while to train.
