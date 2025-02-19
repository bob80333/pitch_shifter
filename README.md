# Pitch Shift Artifact Remover

everything with "_remove_artifacts" is about models that remove pitch shifting artifacts.
they are trained by shifting audio up and then down, which creates audio at same pitch as original, but with all the artifacts of original.
the problem is that they may not generalize to fixing audio at pitches they weren't trained at, since human speech only has a certain range of pitch

everything with "_unshift" is about models that undo a pitch shift.
It is a multi-stage setup
stage 1: train a model that pitch shifts down.  training data is pitch shifted up + original pairs
stage 2: train a model that pitch shifts up.  use the shift down model to produce shifted down, data is shifted down + original pairs.  need shift down model, since normal pitch shift down results in missing top half of frequencies.
stage 3: can use models in a pair e.g. shift up + shift down and train against just original audio, or shift down + shift up.  figure out an additional loss to make sure the models still shift up / down properly instead of just passing audio through.  by doing the paired shifts in opposing order, the idea is that both models learn to produce high quality outputs even if inputs are OOD.
alternatively, repeat stage 1 and 2 but using models trained previously in each repetition, in theory the model produced inputs will approach a better result over time, as the models should be somewhat robust against inputs.  if not, maybe adversarially robust training techniques can be used, or GAN setup