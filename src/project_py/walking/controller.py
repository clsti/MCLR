import sys
sys.path.insert(0, "..")  # noqa
import walking.conf_go2 as conf

import numpy as np
import pinocchio as pin

import crocoddyl


class Go2Controller():

    def __init__(self, model, com_h_ref, integrator="euler", control="zero", fwddyn=True):

        self.model = model
        self.data = model.createData()
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.nv - 6
        self._integrator = integrator
        self._control = control
        self._fwddyn = fwddyn

        self.lfFoot = conf.lfFoot
        self.rfFoot = conf.rfFoot
        self.lhFoot = conf.lhFoot
        self.rhFoot = conf.rhFoot
        self.solver = None 

        # Getting the frame id for all the legs
        self.lfFootId = self.model.getFrameId(self.lfFoot)
        self.rfFootId = self.model.getFrameId(self.rfFoot)
        self.lhFootId = self.model.getFrameId(self.lhFoot)
        self.rhFootId = self.model.getFrameId(self.rhFoot)

        # Defining default state
        q0 = self.model.referenceConfigurations["standing"]
        self.model.defaultState = np.concatenate(
            [q0, np.zeros(self.model.nv)])
        self.com_h_ref = com_h_ref
        # Defining the friction coefficient and normal
        self.mu = 0.7
        self.Rsurf = np.eye(3)

        # State and actuation model
        self.state = crocoddyl.StateMultibody(self.model)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)

        # Running and terminal cost models
        nu = self.actuation.nu if self._fwddyn else self.state.nv
        self.runningCostModel = crocoddyl.CostModelSum(self.state, nu)
        self.terminalCostModel = crocoddyl.CostModelSum(self.state, nu)

    def get_foot_states(self):
        rfFootPos0 = self.data.oMf[self.rfFootId].translation
        rhFootPos0 = self.data.oMf[self.rhFootId].translation
        lfFootPos0 = self.data.oMf[self.lfFootId].translation
        lhFootPos0 = self.data.oMf[self.lhFootId].translation

        return rfFootPos0, rhFootPos0, lfFootPos0, lhFootPos0

    def get_com_ref(self):
        rfFootPos0, rhFootPos0, lfFootPos0, lhFootPos0 = self.get_foot_states()
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = self.com_h_ref
        return comRef

    def solve(self, x0, problem):
        problem = problem(x0)
        self.solver = crocoddyl.SolverBoxDDP(problem)
        self.solver.solve()

        return self.solver.us[0], self.solver.xs[0]

    def get_feedback_gains(self, node_index=0):
        """Get feedback gains k and K from the solved problem"""
        if self.solver is None:
            raise RuntimeError("Must call solve() first")
        
        if not hasattr(self.solver, 'k') or not hasattr(self.solver, 'K'):
            raise AttributeError("Feedback gains not available in solver")
        
        if len(self.solver.k) <= node_index or len(self.solver.K) <= node_index:
            raise IndexError(f"Node index {node_index} out of range")
        
        k = self.solver.k[node_index]
        K = self.solver.K[node_index]
        
        return k, K
    

    def standing_problem(self, x0):
        # action models
        act_models = []

        timeStep = 1e-2
        supportFootIds = [self.lfFootId, self.rfFootId,
                          self.lhFootId, self.rhFootId]

        T = 5
        contact_action = self.create_contact_action(timeStep, supportFootIds)
        for _ in range(T):
            act_models.append(contact_action)

        running_action = act_models[:-1]
        terminal_action = act_models[-1]

        return crocoddyl.ShootingProblem(x0, running_action, terminal_action)

    def create_swingfoot_action(self, timeStep, supportFootIds, comTask, swingFootTask):
        """
        swingFootTask structure: (ID, orientation, position)
        """
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 3 * len(supportFootIds)

        contactModel, costModel = self._create_footswing_model(
            timeStep, supportFootIds, comTask, swingFootTask, nu)

        action_model = self._create_action_model(
            timeStep, contactModel, costModel, nu)

        return action_model

    def _create_footswing_model(self, timeStep, supportFootIds, comTask, swingFootTask, nu):
        contactModel = self._create_contact_model(supportFootIds, nu)
        costModel = crocoddyl.CostModelSum(self.state, nu)
        self._add_contact_cost(costModel, supportFootIds, nu)
        self._add_com_cost(costModel, comTask, nu)
        self._add_swingfoot_cost(costModel, swingFootTask, nu)

        return contactModel, costModel

    def create_contact_action(self, timeStep, supportFootIds):
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 3 * len(supportFootIds)

        # get contact and cost model
        contactModel = self._create_contact_model(supportFootIds, nu)
        costModel = crocoddyl.CostModelSum(self.state, nu)
        self._add_contact_cost(costModel, supportFootIds, nu)
        # add gravity cost
        self._add_gravity_compensation_cost(costModel, 1e2)

        # create action model
        action_model = self._create_action_model(
            timeStep, contactModel, costModel, nu)

        return action_model

    def _create_action_model(self, timeStep, contactModel, costModel, nu):
        """Create Action model for KKT conditions"""
        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contactModel, costModel, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contactModel, costModel
            )
        if self._control == "one":
            control = crocoddyl.ControlParametrizationModelPolyOne(nu)
        elif self._control == "rk4":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(
                nu, crocoddyl.RKType.four
            )
        elif self._control == "rk3":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(
                nu, crocoddyl.RKType.three
            )
        else:
            control = crocoddyl.ControlParametrizationModelPolyZero(nu)
        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(
                dmodel, control, timeStep)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, control, crocoddyl.RKType.four, timeStep
            )
        elif self._integrator == "rk3":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, control, crocoddyl.RKType.three, timeStep
            )
        elif self._integrator == "rk2":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, control, crocoddyl.RKType.two, timeStep
            )
        else:
            model = crocoddyl.IntegratedActionModelEuler(
                dmodel, control, timeStep)
        return model

    def _create_contact_model(self, supportFootIds, nu):
        """Create a 3D multi-contact model including supporting"""
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel3D(
                self.state,
                i,
                np.array([0.0, 0.0, 0.0]),  # local offset
                pin.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),  # force limits
            )
            contactModel.addContact(
                self.model.frames[i].name + "_contact", supportContactModel
            )
        return contactModel

    def _add_swingfoot_cost(self, costModel, swingFootTask, nu):
        """
        swingFootTask structure: (ID, orientation, position)
        """
        for i in swingFootTask:
            frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
                self.state, i[0], i[1].translation, nu
            )
            footTrack = crocoddyl.CostModelResidual(
                self.state, frameTranslationResidual
            )
            costModel.addCost(
                self.rmodel.frames[i[0]].name +
                "_footTrack", footTrack, 1e6
            )

    def _add_contact_cost(self, costModel, supportFootIds, nu):
        """Creating the cost model for a contact phase"""
        # Friction model
        for i in supportFootIds:
            cone = crocoddyl.FrictionCone(self.Rsurf, self.mu, 4, False)
            coneResidual = crocoddyl.ResidualModelContactFrictionCone(
                self.state, i, cone, nu, self._fwddyn
            )
            coneActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            frictionCone = crocoddyl.CostModelResidual(
                self.state, coneActivation, coneResidual
            )
            costModel.addCost(
                self.model.frames[i].name + "_frictionCone", frictionCone, 1e1
            )
        stateWeights = np.array(
            [0.0] * 3                       # base position (x, y, z)
            + [500.0] * 3                   # base orientation (roll, pitch, yaw) # noqa
            + [0.01] * (self.model.nv - 6)  # joint positions
            + [10.0] * 6                    # base linear & angular velocity
            + [1.0] * (self.model.nv - 6)   # joint velocities
        )
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.model.defaultState, nu
        )
        stateActivation = crocoddyl.ActivationModelWeightedQuad(
            stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )
        if self._fwddyn:
            ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
            ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        else:
            ctrlResidual = crocoddyl.ResidualModelJointEffort(
                self.state, self.actuation, nu
            )
            ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e4)

        # Bound cost
        lb = np.concatenate(
            [self.state.lb[1: self.state.nv + 1], self.state.lb[-self.state.nv:]]
        )
        ub = np.concatenate(
            [self.state.ub[1: self.state.nv + 1], self.state.ub[-self.state.nv:]]
        )
        stateBoundsResidual = crocoddyl.ResidualModelState(self.state, nu)
        stateBoundsActivation = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(lb, ub)
        )
        stateBounds = crocoddyl.CostModelResidual(
            self.state, stateBoundsActivation, stateBoundsResidual
        )
        costModel.addCost("stateBounds", stateBounds, 1e3)

    def _add_com_cost(self, costModel, com_d, nu):
        com_residual = crocoddyl.ResidualModelCoMPosition(
            self.state, com_d, nu)
        comTrack = crocoddyl.CostModelResidual(self.state, com_residual)
        costModel.addCost("comTrack", comTrack, 1e6)

    def _add_gravity_compensation_cost(self, costModel, weight=1e1):
        """Adds gravity compensation cost model"""
        nu = self.actuation.nu if self._fwddyn else self.nv
        uResidual = crocoddyl.ResidualModelContactControlGrav(self.state, nu)
        activation = crocoddyl.ActivationModelQuad(self.nv)
        ctrlReg = crocoddyl.CostModelResidual(
            self.state, activation, uResidual)
        costModel.addCost("gravityComp", ctrlReg, weight)

    def _add_com_height_cost(self, costModel, height_target=0.335, weight=1e2):
        com_ref = np.array([0., 0., height_target])
        comResidual = crocoddyl.ResidualModelCoMPosition(
            self.state, com_ref, self.actuation.nu)
        comActivation = crocoddyl.ActivationModelWeightedQuad(
            np.array([1., 1., 10.]))
        comCost = crocoddyl.CostModelResidual(
            self.state, comActivation, comResidual)
        costModel.addCost("comHeight", comCost, weight)

    def createFootSwitchModel(self, supportFootIds, swingFootTask, pseudoImpulse=False):
        """Action model for a foot switch phase.

        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :param pseudoImpulse: true for pseudo-impulse models, otherwise it uses the
            impulse model
        :return action model for a foot switch phase
        """
        if pseudoImpulse:
            return self.createPseudoImpulseModel(supportFootIds, swingFootTask)
        else:
            return self.createImpulseModel(supportFootIds, swingFootTask)

    def createPseudoImpulseModel(self, supportFootIds, swingFootTask):
        """Action model for pseudo-impulse models.

        A pseudo-impulse model consists of adding high-penalty cost for the contact
        velocities.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return pseudo-impulse differential action model
        """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 3 * len(supportFootIds)
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel3D(
                self.state,
                i,
                np.array([0.0, 0.0, 0.0]),
                pin.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),
            )
            contactModel.addContact(
                self.rmodel.frames[i].name + "_contact", supportContactModel
            )
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        for i in supportFootIds:
            cone = crocoddyl.FrictionCone(self.Rsurf, self.mu, 4, False)
            coneResidual = crocoddyl.ResidualModelContactFrictionCone(
                self.state, i, cone, nu, self._fwddyn
            )
            coneActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            frictionCone = crocoddyl.CostModelResidual(
                self.state, coneActivation, coneResidual
            )
            costModel.addCost(
                self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e1
            )
        if swingFootTask is not None:
            for i in swingFootTask:
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
                    self.state, i[0], i[1].translation, nu
                )
                frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(
                    self.state,
                    i[0],
                    pin.Motion.Zero(),
                    pin.LOCAL_WORLD_ALIGNED,
                    nu,
                )
                footTrack = crocoddyl.CostModelResidual(
                    self.state, frameTranslationResidual
                )
                impulseFootVelCost = crocoddyl.CostModelResidual(
                    self.state, frameVelocityResidual
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name +
                    "_footTrack", footTrack, 1e7
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name + "_impulseVel",
                    impulseFootVelCost,
                    1e6,
                )
        stateWeights = np.array(
            [0.0] * 3
            + [500.0] * 3
            + [0.01] * (self.rmodel.nv - 6)
            + [10.0] * self.rmodel.nv
        )
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.rmodel.defaultState, nu
        )
        stateActivation = crocoddyl.ActivationModelWeightedQuad(
            stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )
        if self._fwddyn:
            ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
            ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        else:
            ctrlResidual = crocoddyl.ResidualModelJointEffort(
                self.state, self.actuation, nu
            )
            ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-3)
        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contactModel, costModel, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contactModel, costModel
            )
        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.0)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, crocoddyl.RKType.four, 0.0
            )
        elif self._integrator == "rk3":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, crocoddyl.RKType.three, 0.0
            )
        elif self._integrator == "rk2":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, crocoddyl.RKType.two, 0.0)
        else:
            model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.0)
        return model

    def createImpulseModel(
        self, supportFootIds, swingFootTask, JMinvJt_damping=1e-12, r_coeff=0.0
    ):
        """Action model for impulse models.

        An impulse model consists of describing the impulse dynamics against a set of
        contacts.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return impulse action model
        """
        # Creating a 3D multi-contact model, and then including the supporting foot
        impulseModel = crocoddyl.ImpulseModelMultiple(self.state)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ImpulseModel3D(
                self.state, i, pin.LOCAL_WORLD_ALIGNED
            )
            impulseModel.addImpulse(
                self.rmodel.frames[i].name + "_impulse", supportContactModel
            )
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, 0)
        if swingFootTask is not None:
            for i in swingFootTask:
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
                    self.state, i[0], i[1].translation, 0
                )
                footTrack = crocoddyl.CostModelResidual(
                    self.state, frameTranslationResidual
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name +
                    "_footTrack", footTrack, 1e7
                )
        stateWeights = np.array(
            [1.0] * 6 + [10.0] * (self.rmodel.nv - 6) + [10.0] * self.rmodel.nv
        )
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.rmodel.defaultState, 0
        )
        stateActivation = crocoddyl.ActivationModelWeightedQuad(
            stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )
        costModel.addCost("stateReg", stateReg, 1e1)
        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        model = crocoddyl.ActionModelImpulseFwdDynamics(
            self.state, impulseModel, costModel
        )
        model.JMinvJt_damping = JMinvJt_damping
        model.r_coeff = r_coeff
        return model
