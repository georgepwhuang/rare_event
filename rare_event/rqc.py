import pennylane as qml
import math

def MarkovianRecurrentQuantumCircuit(wires, memory_state_prep_list, transition, initial_state=0):
    mem_bits = int(math.log2(len(memory_state_prep_list[initial_state])))
    in_bits = int(math.log2(len(transition))) - mem_bits
    assert (len(wires) - mem_bits) % in_bits == 0
    layers = (len(wires) - mem_bits) // in_bits
    qml.StatePrep(memory_state_prep_list[initial_state], wires=wires[:mem_bits])
    for i in range(layers):
        qml.QubitUnitary(transition, wires=wires[:mem_bits]+wires[mem_bits+i*in_bits:mem_bits+(i+1)*in_bits])
    for i, memory_prep in enumerate(memory_state_prep_list):
        qml.ctrl(qml.adjoint(qml.StatePrep), control=wires[mem_bits+(layers-1)*in_bits:mem_bits+layers*in_bits], control_values=[x == '1' for x in "{0:0{1}b}".format(i,in_bits)])(memory_prep, wires=wires[:mem_bits])