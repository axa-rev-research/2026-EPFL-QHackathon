import quboforpennylane as qml
dev = qml.device("lightning.gpu", wires=2)
print("GPU OK!")