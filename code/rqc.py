import pennylane as qml
import math

def RecurrentQuantumCircuit(wires, memory_state_prep, transition):
    mem_bits = int(math.log2(len(memory_state_prep)))
    in_bits = int(math.log2(len(transition))) - mem_bits
    assert (len(wires) - mem_bits) % in_bits == 0
    layers = (len(wires) - mem_bits) // in_bits
    qml.StatePrep(memory_state_prep, wires=wires[:mem_bits])
    for i in range(layers):
        qml.QubitUnitary(transition, wires=wires[:mem_bits]+wires[mem_bits+i*in_bits:mem_bits+(i+1)*in_bits])
    