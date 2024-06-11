import os
import pickle
import numpy as np
import openmdao.api as om

class Battery(om.ExplicitComponent):

    def setup(self):
        self.add_input('voltage_supply', units = 'V')
        self.add_input('current', units = 'A')
        self.add_input('resistance', units = 'ohm')

        self.add_output('voltage_out', units = 'V')
        self.add_output('power', units = 'W')

        self.declare_partials(['voltage_out', 'power'], ['voltage_supply', 'current', 'resistance'])

    def compute(self, inputs, outputs):
        outputs['voltage_out'] = inputs['voltage_supply'] - inputs['current'] * inputs['resistance']
        outputs['power'] = inputs['current'] * inputs['voltage_supply'] - inputs['current']**2 * inputs['resistance']

    def compute_partials(self, inputs, partials):
        partials['voltage_out', 'voltage_supply'] = 1
        partials['voltage_out', 'current'] = -inputs['resistance']
        partials['voltage_out', 'resistance'] = -inputs['current']

        partials['power', 'voltage_supply'] = inputs['current']
        partials['power', 'current'] = inputs['voltage_supply'] - 2 * inputs['current'] * inputs['resistance']
        partials['power', 'resistance'] = -inputs['current']**2

class ElectronicSpeedController(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('a', default = 1.6054, desc = 'a coefficient for efficiency(throttle) equation: efficiency = a * (1 - 1 / (1 + c*throttle^d))')
        self.options.declare('b', default = 1.6519, desc = 'b coefficient for efficiency(throttle) equation: efficiency = a * (1 - 1 / (1 + c*throttle^d))')
        self.options.declare('c', default = 0.6455, desc = 'c coefficient for efficiency(throttle) equation: efficiency = a * (1 - 1 / (1 + c*throttle^d))')

    def setup(self):
        self.add_input('voltage_in', units = 'V')
        self.add_input('current_in', units = 'A')
        self.add_input('throttle')

        self.add_output('efficiency')
        self.add_output('voltage_out', units = 'V')
        self.add_output('current_out', units = 'A')
        self.add_output('power', units = 'W')

        self.declare_partials('efficiency', 'throttle')
        self.declare_partials('voltage_out', ['voltage_in', 'throttle'])
        self.declare_partials('current_out', ['current_in', 'throttle'])
        self.declare_partials('power', ['voltage_in', 'current_in', 'throttle'])

    def compute(self, inputs, outputs):
        
        a = self.options['a']
        b = self.options['b']
        c = self.options['c']
        outputs['efficiency'] = a * (1 - 1 / (1 + b*inputs['throttle']**c))

        outputs['voltage_out'] = inputs['voltage_in'] * inputs['throttle'] * outputs['efficiency']
        outputs['current_out'] = inputs['current_in'] / inputs['throttle']
        outputs['power'] = (outputs['efficiency'] - 1) * inputs['current_in'] * inputs['voltage_in']

    def compute_partials(self, inputs, partials):

        a = self.options['a']
        b = self.options['b']
        c = self.options['c']
        t = inputs['throttle']
        efficiency = a * (1 - 1 / (1 + b*t**c))
        partials['efficiency', 'throttle'] = a*b*c*t**(c - 1) / (b*t**c + 1)**2

        partials['voltage_out', 'voltage_in'] = inputs['throttle'] * efficiency
        partials['voltage_out', 'throttle'] = inputs['voltage_in'] * (efficiency + inputs['throttle'] * partials['efficiency', 'throttle'])

        partials['current_out', 'current_in'] = 1 / inputs['throttle']
        partials['current_out', 'throttle'] = -inputs['current_in'] / inputs['throttle']**2

        partials['power', 'voltage_in'] = (efficiency - 1) * inputs['current_in']
        partials['power', 'current_in'] = (efficiency - 1) * inputs['voltage_in']
        partials['power', 'throttle'] = inputs['current_in'] * inputs['voltage_in'] * partials['efficiency', 'throttle']

class Motor(om.ExplicitComponent):

    def setup(self):
        self.add_input('voltage_in', units = 'V')
        self.add_input('current', units = 'A')
        self.add_input('resistance', units = 'ohm')
        self.add_input('kv', units = 'rpm / V')
        self.add_input('idle_current', units = 'A')
        
        self.add_output('rpm', units = 'rpm')
        self.add_output('power', units = 'W')

        self.declare_partials(['rpm', 'power'], ['voltage_in', 'current', 'resistance', 'kv', 'idle_current'])

    def compute(self, inputs, outputs):
        voltage_prop = inputs['voltage_in'] - inputs['current'] * inputs['resistance']
        outputs['rpm'] = inputs['kv'] * voltage_prop
        outputs['power'] = -inputs['current']**2 * inputs['resistance'] - inputs['idle_current'] * voltage_prop

    def compute_partials(self, inputs, partials):
        voltage_prop = inputs['voltage_in'] - inputs['current'] * inputs['resistance']
        dvoltage_prop_dvoltage_in = 1
        dvoltage_prop_dcurrent = -inputs['resistance']
        dvoltage_prop_dresistance = -inputs['current']

        partials['rpm', 'voltage_in'] = inputs['kv'] * dvoltage_prop_dvoltage_in
        partials['rpm', 'current'] = inputs['kv'] * dvoltage_prop_dcurrent
        partials['rpm', 'resistance'] = inputs['kv'] * dvoltage_prop_dresistance
        partials['rpm', 'kv'] = voltage_prop
        partials['rpm', 'idle_current'] = 0

        partials['power', 'voltage_in'] = -inputs['idle_current'] * dvoltage_prop_dvoltage_in
        partials['power', 'current'] = -2 * inputs['current'] * inputs['resistance'] - inputs['idle_current'] * dvoltage_prop_dcurrent
        partials['power', 'resistance'] = -inputs['current']**2 - inputs['idle_current'] * dvoltage_prop_dresistance
        partials['power', 'kv'] = 0
        partials['power', 'idle_current'] = -voltage_prop

class Propeller(om.MetaModelUnStructuredComp):
    pass

class PowerNet(om.ImplicitComponent):

    def setup(self):
        self.add_input('power_batt', units = 'W')
        self.add_input('power_esc', units = 'W')
        self.add_input('power_motor', units = 'W')
        self.add_input('power_prop', units = 'W')

        self.add_output('current', units = 'A')

        self.add_residual('power_net', units = 'W')

        self.declare_partials('power_net', ['power_batt', 'power_esc', 'power_motor', 'power_prop'], val = 1)

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['power_net'] = inputs['power_batt'] + inputs['power_esc'] + inputs['power_motor'] + inputs['power_prop']

class ElectricPropulsion(om.Group):

    def setup(self):

        dirSurrogateModels = os.path.join(os.path.dirname(__file__), 'surrogate_models')
        dirPropModel = os.path.join(dirSurrogateModels, 'prop_model')
        pathSurrogateModelData = os.path.join(dirPropModel, 'surrogate_model_data.pickle')
        pathThrustTrainingData = os.path.join(dirPropModel, 'thrust_training_data.dat')
        pathPowerTrainingData = os.path.join(dirPropModel, 'power_training_data.dat')

        with open(pathSurrogateModelData, 'rb') as fileSurrogateModelData:
            surrogateModelData = pickle.load(fileSurrogateModelData)

        propModel = om.MetaModelUnStructuredComp()

        propModel.add_input('diameter', training_data = surrogateModelData['diameter'], units = 'inch')
        propModel.add_input('pitch', training_data = surrogateModelData['pitch'], units = 'inch')
        propModel.add_input('rpm', training_data = surrogateModelData['rpm'], units = 'rpm')
        propModel.add_input('velocity', training_data = surrogateModelData['velocity'], units = 'm/s')

        propModel.add_output('thrust', 0.0, training_data = surrogateModelData['thrust'], surrogate = om.KrigingSurrogate(eval_rmse = True, lapack_driver = 'gesdd', training_cache = pathThrustTrainingData), units = 'N')
        propModel.add_output('power', 0.0, training_data = surrogateModelData['power'], surrogate = om.KrigingSurrogate(eval_rmse = True, lapack_driver = 'gesdd', training_cache = pathPowerTrainingData), units = 'W')

        self.add_subsystem('battery', Battery())
        self.add_subsystem('esc', ElectronicSpeedController())
        self.add_subsystem('motor', Motor())
        self.add_subsystem('prop', propModel)
        self.add_subsystem('power_net', PowerNet())

        self.connect('battery.voltage_out', 'esc.voltage_in')
        self.connect('esc.voltage_out', 'motor.voltage_in')
        self.connect('esc.current_out', 'motor.current')
        self.connect('motor.rpm', 'prop.rpm')

        self.connect('battery.power', 'power_net.power_batt')
        self.connect('esc.power', 'power_net.power_esc')
        self.connect('motor.power', 'power_net.power_motor')
        self.connect('prop.power', 'power_net.power_prop')
        self.connect('power_net.current', ['battery.current', 'esc.current_in'])

class RubberMotor(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('a_io', default =  0.3040, desc = 'a coefficient for idle_current(kv, mass) equation: idle_current = kv^a * mass^b + c')
        self.options.declare('b_io', default =  0.2409, desc = 'b coefficient for idle_current(kv, mass) equation: idle_current = kv^a * mass^b + c')
        self.options.declare('c_io', default = -3.6401, desc = 'c coefficient for idle_current(kv, mass) equation: idle_current = kv^a * mass^b + c')

        self.options.declare('a_r', default =  0.003356, desc = 'a coefficient for resistance(mass, idle_current) equation: resistance = a/mass + b/idle_current + c')
        self.options.declare('b_r', default =  0.04760,  desc = 'b coefficient for resistance(mass, idle_current) equation: resistance = a/mass + b/idle_current + c')
        self.options.declare('c_r', default = -0.02631,  desc = 'c coefficient for resistance(mass, idle_current) equation: resistance = a/mass + b/idle_current + c')

        self.options.declare('a_pow', default =  -0.1316,     desc = 'a coefficient for max_power(kv, mass) equation: max_power = a*kv + b*mass + c*kv*mass + d')
        self.options.declare('b_pow', default =   5.5451e+03, desc = 'b coefficient for max_power(kv, mass) equation: max_power = a*kv + b*mass + c*kv*mass + d')
        self.options.declare('c_pow', default =  -2.0421,     desc = 'c coefficient for max_power(kv, mass) equation: max_power = a*kv + b*mass + c*kv*mass + d')
        self.options.declare('d_pow', default = 181.3208,     desc = 'd coefficient for max_power(kv, mass) equation: max_power = a*kv + b*mass + c*kv*mass + d')

    def setup(self):
        self.add_input('kv', units = 'rpm/V')
        self.add_input('mass', units = 'kg')
        
        self.add_output('resistance', units = 'ohm')
        self.add_output('idle_current', units = 'A')
        self.add_output('max_power', units = 'W')
        self.add_output('kv_out', units = 'rpm/V')

        self.declare_partials(['resistance', 'idle_current', 'max_power'], ['kv', 'mass'])
        self.declare_partials('kv_out', 'kv')
    
    def compute(self, inputs, outputs):

        a_io = self.options['a_io']
        b_io = self.options['b_io']
        c_io = self.options['c_io']

        a_r = self.options['a_r']
        b_r = self.options['b_r']
        c_r = self.options['c_r']

        a_pow = self.options['a_pow']
        b_pow = self.options['b_pow']
        c_pow = self.options['c_pow']
        d_pow = self.options['d_pow']

        outputs['idle_current'] = inputs['kv']**a_io * inputs['mass']**b_io + c_io
        outputs['resistance'] = a_r / inputs['mass'] + b_r / outputs['idle_current'] + c_r
        outputs['max_power'] = a_pow*inputs['kv'] + b_pow*inputs['mass'] + c_pow*inputs['kv']*inputs['mass'] + d_pow
        outputs['kv_out'] = inputs['kv']
    
    def compute_partials(self, inputs, partials):

        a_io = self.options['a_io']
        b_io = self.options['b_io']
        c_io = self.options['c_io']

        a_r = self.options['a_r']
        b_r = self.options['b_r']
        c_r = self.options['c_r']

        a_pow = self.options['a_pow']
        b_pow = self.options['b_pow']
        c_pow = self.options['c_pow']
        d_pow = self.options['d_pow']

        idle_current = inputs['kv']**a_io * inputs['mass']**b_io + c_io
        partials['idle_current', 'kv'] = a_io*inputs['kv']**(a_io - 1) * inputs['mass']**b_io
        partials['idle_current', 'mass'] = inputs['kv']**a_io * b_io*inputs['mass']**(b_io - 1)
        
        partials['resistance', 'kv'] = -b_r/idle_current**2 * partials['idle_current', 'kv']
        partials['resistance', 'mass'] = -a_r/inputs['mass']**2 - b_r/idle_current**2 * partials['idle_current', 'mass']

        partials['max_power', 'kv'] = a_pow + c_pow * inputs['mass']
        partials['max_power', 'mass'] = b_pow + c_pow * inputs['kv']

        partials['kv_out', 'kv'] = 1

class RubberElectricPropulsion(om.Group):

    def setup(self):

        self.add_subsystem('electric_propulsion', ElectricPropulsion())
        self.add_subsystem('rubber_motor', RubberMotor())

        self.connect('rubber_motor.kv_out', 'electric_propulsion.motor.kv')
        self.connect('rubber_motor.resistance', 'electric_propulsion.motor.resistance')
        self.connect('rubber_motor.idle_current', 'electric_propulsion.motor.idle_current')

if __name__=="__main__":

    '''
    prob = om.Problem()
    model = prob.model
    model.nonlinear_solver = om.NewtonSolver(solve_subsystems = True, iprint = 2)
    model.linear_solver = om.DirectSolver(iprint = 2)
    model.add_subsystem('electric_propulsion', ElectricPropulsion(), promotes=['*'])

    prob.setup()

    prob.set_val('battery.voltage_supply', 22.2, units = 'V')
    prob.set_val('battery.resistance', 0.012, units = 'ohm')
    prob.set_val('esc.throttle', 1)
    prob.set_val('motor.kv', 280, units = 'rpm/V')
    prob.set_val('motor.idle_current', 1.2, units = 'A')
    prob.set_val('motor.resistance', 26.3, units = 'mohm')
    prob.set_val('prop.diameter', 22, units = 'inch')
    prob.set_val('prop.pitch', 10, units = 'inch')
    prob.set_val('prop.velocity', 0, units = 'm/s')
    prob.set_val('power_net.current', 0, units = 'A')

    prob.run_model()
    '''

    prob = om.Problem()
    model = prob.model
    model.nonlinear_solver = om.NewtonSolver(solve_subsystems = True, iprint = 2)
    model.linear_solver = om.DirectSolver(iprint = 2)
    model.add_subsystem('rubber_electric_propulsion', RubberElectricPropulsion(), promotes=['*'])

    prob.setup()

    prob.set_val('rubber_motor.kv', 280, units = 'rpm/V')
    prob.set_val('rubber_motor.mass', 500, units = 'g')

    prob.set_val('electric_propulsion.battery.voltage_supply', 22.2, units = 'V')
    prob.set_val('electric_propulsion.battery.resistance', 0.012, units = 'ohm')
    prob.set_val('electric_propulsion.esc.throttle', 1)
    prob.set_val('electric_propulsion.prop.diameter', 22, units = 'inch')
    prob.set_val('electric_propulsion.prop.pitch', 10, units = 'inch')
    prob.set_val('electric_propulsion.prop.velocity', 0, units = 'm/s')
    prob.set_val('electric_propulsion.power_net.current', 0, units = 'A')

    prob.run_model()
    a = 5

    # om.n2(p)
