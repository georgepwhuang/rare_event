import pennylane as qml

def ProjCtrlPhaseShift(control_wires, target_wire, phi):
    qml.MultiControlledX(wires=(*control_wires, target_wire), 
                         control_values=[0]*len(control_wires))
    qml.RZ(phi = 2*phi, wires=target_wire)
    qml.MultiControlledX(wires=(*control_wires, target_wire), 
                         control_values=[0]*len(control_wires))
    
def QET(BE, wires, ancilla_wires, angles, 
        simulate=True, control_wires=None, 
        rotation_wire=None, *args, **kwargs):
    deg = len(angles)
    qml.Hadamard(rotation_wire)
    if control_wires is None:
        control_wires = ancilla_wires
    ProjCtrlPhaseShift(control_wires=control_wires, 
                       target_wire=rotation_wire, 
                       phi=angles[-1])
    for i in range(1, deg):
        BE(wires=wires, 
           ancilla_wires=ancilla_wires, 
           simulate=simulate, *args, **kwargs)
        ProjCtrlPhaseShift(control_wires=control_wires, 
                           target_wire=rotation_wire, 
                           phi=angles[-i-1])
    qml.Hadamard(rotation_wire)