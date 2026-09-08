# TODO

## High Priority
- [ ] 

## Low Priority
- [x] Use shipped mace run_train, instead of manually requiring to provide a path to the script
- [x] Add util to automatically generate mace-jax foundation model. torch model could be pulled via mace API. Would make things more accessible
- [x] add a CLI flag to print which deps are installed (could use _capabilities)
- [x] rework NeuralILTrainer, maybe also use yaml instead of JSON
- [ ] Think about reworking all config files from JSON to yaml, could probably be done backwards-compatible
- [ ] Option to pass VASPHandler INCAR kwargs also via a yaml file
- [x] Give the RemoteTask derived objects the possibility to finish_check, BEFORE submitting, to prevent redoing calculations that were already finished (for crash scenarios)
- [x] In RemoteTrajectoryProcessor allow shard_size to be passed as -1 or None, which means that the entire set of atoms is processed without sharding
