import os
import pickle
import time
import numpy as np
import scipy as sc
import openmdao.api as om
import matplotlib.pyplot as plt
from openproprc import ElectricPropulsion

'''
class MotorPropeller(om.ExplicitComponent):

    def setup(self):
        self.add_input('motorModel', val = None)
        self.add_input('propDiameter', val = 0.0, units = 'in')
        self.add_input('propPitch', val = 0.0, units = 'in')
        self.add_input('throttle', val = 0.0)
        self.add_input('velocity', units = 'm/s')

        self.add_output('thrust', val = 0.0, units = 'N')
        self.add_output('inputPower', val = 0.0, units = 'W')

    def setup_partials(self):
        self.declare_partials('thrust', ['propDiameter', 'propPitch', 'throttle', 'velocity'])
        self.declare_partials('inputPower', ['propDiameter', 'propPitch', 'throttle', 'velocity'])
    
    def compute(self, inputs, outputs):
'''

# plt.rcParams["figure.autolayout"] = True

prob = om.Problem()
model = prob.model
model.nonlinear_solver = om.NewtonSolver(solve_subsystems = False)
model.linear_solver = om.DirectSolver()
model.add_subsystem('electric_propulsion', ElectricPropulsion(), promotes = ['*'])
model.add_design_var('esc.throttle', lower = 0, upper = 1)
model.add_objective('prop.thrust', scaler = -1)
model.add_constraint('battery.power', upper = 1000, units = 'W')

prob.driver = om.pyOptSparseDriver(optimizer = 'IPOPT')
# prob.driver.options['disp'] = True
prob.setup()

prob.set_val('battery.voltage_supply', 22.2, units = 'V')
prob.set_val('battery.resistance', 0.012, units = 'ohm')
prob.set_val('motor.kv', 280, units = 'rpm / V')
prob.set_val('motor.idle_current', 1.2, units = 'A')
prob.set_val('motor.resistance', 26.3, units = 'mohm')
prob.set_val('prop.diameter', 22, units = 'inch')
prob.set_val('prop.pitch', 10, units = 'inch')

vMax = 60
n = 51
velocity = np.linspace(0, vMax, n)

thrustFullThrottle = np.zeros(n)
thrustPowerLimited = np.zeros(n)
powerFullThrottle = np.zeros(n)
efficiencyFullThrottle = np.zeros(n)

powerPowerLimited = np.zeros(n)
throttleFullThrottle = np.zeros(n)
throttlePowerLimited = np.zeros(n)
efficiencyPowerLimited = np.zeros(n)

for idxV, v in enumerate(velocity):
    prob.set_val('esc.throttle', 1)
    prob.set_val('prop.velocity', v, units = 'mi / h')
    prob.set_val('power_net.current', 10, units = 'A')
    prob.run_model()
    thrustFullThrottle[idxV] = prob.get_val('prop.thrust', units = 'lbf')
    powerFullThrottle[idxV] = prob.get_val('battery.power', units = 'W')
    throttleFullThrottle[idxV] = prob.get_val('esc.throttle')
    efficiencyFullThrottle[idxV] = prob.get_val('prop.thrust', units = 'N') * prob.get_val('prop.velocity', units = 'm / s') / (prob.get_val('battery.voltage_supply', units = 'V') * prob.get_val('battery.current', units = 'A'))

    '''
    if velocity[idxV] > 58.0:
        
        n = 21
        throttleAtVelocity = np.linspace(0, 1, n)
        thrustAtVelocity = np.zeros(n)
        powerAtVelocity = np.zeros(n)
        for idxT, t in enumerate(throttleAtVelocity):
            prob.set_val('motorModel.throttle', t)
            prob.run_model()
            thrustAtVelocity[idxT] = prob.get_val('motorModel.thrust')*0.224809
            powerAtVelocity[idxT] = prob.get_val('motorModel.inputPower')

        fig = plt.figure()
        grid = fig.add_gridspec(2, hspace = 0)
        axes = grid.subplots(sharex = True)
        fig.suptitle(f'{motor}, {velocity[idxV]:.2f} mph')

        y = [thrustAtVelocity, powerAtVelocity]
        yLabels = ['Thrust (lb)', 'Power (W)']

        for idxPlot, ax in enumerate(axes):
            axes[idxPlot].plot(throttleAtVelocity, y[idxPlot])
            axes[idxPlot].set(xlabel = 'Throttle', ylabel = yLabels[idxPlot], xlim = [0, 1])
            axes[idxPlot].label_outer()

        plt.show()
    '''

    prob.run_driver()
    
    thrustPowerLimited[idxV] = prob.get_val('prop.thrust', units = 'lbf')
    powerPowerLimited[idxV] = prob.get_val('battery.power', units = 'W')
    throttlePowerLimited[idxV] = prob.get_val('esc.throttle')
    efficiencyPowerLimited[idxV] = prob.get_val('prop.thrust', units = 'N') * prob.get_val('prop.velocity', units = 'm / s') / (prob.get_val('battery.voltage_supply', units = 'V') * prob.get_val('battery.current', units = 'A'))

fig = plt.figure()
grid = fig.add_gridspec(4, hspace = 0.5)
axes = grid.subplots(sharex = True)
# fig.suptitle(motor)

yFullThrottle = [thrustFullThrottle, powerFullThrottle, throttleFullThrottle, efficiencyFullThrottle]
yPowerLimited = [thrustPowerLimited, powerPowerLimited, throttlePowerLimited, efficiencyPowerLimited]
yLims = [None, None, 1, 1]
yLabels = ['Thrust (lb)', 'Power (W)', 'Throttle', 'Overall Efficiency']

for idxPlot, ax in enumerate(axes):
    ax.plot(velocity, yFullThrottle[idxPlot])
    ax.plot(velocity, yPowerLimited[idxPlot])
    ax.set(xlabel = 'Velocity (mph)', ylabel = yLabels[idxPlot], xlim = [0, vMax], ylim = [0, yLims[idxPlot]])
    ax.label_outer()
    ax.grid(visible = True)

plt.show()
