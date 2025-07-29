import sys
sys.path.insert(0, "..")  # noqa
import walking.conf_go2 as conf

import numpy as np
import pinocchio as pin
import crocoddyl


class Go2Controller():
    """
    Go2Controller class for controlling a Go2 quadruped robot using Crocoddyl.
    """

    def __init__(self, model, data, com_h_ref, integrator="euler", control="zero", fwddyn=True):
        """
        Args:
            model: Robot model 
            data: Robot data
            com_h_ref: Desired center of mass height
            integrator: type of the integrator
                (options are: 'euler', and 'rk4')
            control: type of control parametrization
                (options are: 'zero', 'one', and 'rk4')
            fwddyn: True for forward-dynamics and False for inverse-dynamics
                formulations
        """
        self.model = model
        self.data = data
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
        self.mu = 0.4
        self.Rsurf = np.eye(3)

        # flag for first step
        self.firstStep = True

        # State and actuation model
        self.state = crocoddyl.StateMultibody(self.model)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)

        # Running and terminal cost models
        nu = self.actuation.nu if self._fwddyn else self.state.nv
        self.runningCostModel = crocoddyl.CostModelSum(self.state, nu)
        self.terminalCostModel = crocoddyl.CostModelSum(self.state, nu)

    #################################################################################
    # ----------------------------------- Solver ---------------------------------- #
    #################################################################################

    def solve(self, x0, problem, xs_init=None, us_init=None):
        """
        Solve the optimal control problem.

        Args:
            x0: Initial state
            problem: Crocoddyl shooting problem
            xs_init: Initial guess for state trajectory
            us_init: Initial guess for control trajectory

        Returns:
            tuple: (controls, states)
        """
        self.solver = crocoddyl.SolverBoxDDP(problem)

        T = self.solver.problem.T

        # Use initial guess for warmstart
        if us_init is not None and len(us_init) == T:
            us = us_init
        else:
            us = self.solver.problem.quasiStatic([x0] * T)

        if xs_init is not None and len(xs_init) == T + 1:
            xs = xs_init
        else:
            xs = [x0] * (T + 1)

        self.solver.solve(xs, us, 100, False, 0.1)
        return self.solver.us, self.solver.xs

    def get_feedback_gains(self, node_index=0):
        """
        Get feedback gains k and K from the solved problem
        """
        if self.solver is None:
            raise RuntimeError("Must call solve() first")

        if not hasattr(self.solver, 'k') or not hasattr(self.solver, 'K'):
            raise AttributeError("Feedback gains not available in solver")

        if len(self.solver.k) <= node_index or len(self.solver.K) <= node_index:
            raise IndexError(f"Node index {node_index} out of range")

        k = self.solver.k[node_index]
        K = self.solver.K[node_index]

        return k, K

    #################################################################################
    # ---------------------------------- Problems --------------------------------- #
    #################################################################################

    def standing_problem(self, x0):
        """
        Create a simple standing (static) Crocoddyl problem with full contact.

        Args:
            x0 (np.ndarray): Initial robot state.

        Returns:
            crocoddyl.ShootingProblem: Shooting problem with standing contact.
        """
        # action models
        act_models = []
        timeStep = 1e-2
        supportFootIds = [self.lfFootId, self.rfFootId,
                          self.lhFootId, self.rhFootId]

        T = 5  # Horizon length
        contact_action = self.create_contact_action(timeStep, supportFootIds)
        for _ in range(T):
            act_models.append(contact_action)

        running_action = act_models[:-1]
        terminal_action = act_models[-1]

        return crocoddyl.ShootingProblem(x0, running_action, terminal_action)

    def walking_problem_ocp(self, x0, timeStep, foot_traj, com_traj):
        """
        Create a full walking optimal control problem (OCP).

        Args:
            x0 (np.ndarray): Initial state.
            timeStep (float): Time step duration.
            foot_traj (list): List of foot trajectories for each phase.
            com_traj (list): List of CoM trajectories for each phase.

        Returns:
            crocoddyl.ShootingProblem: A shooting problem representing the walking motion.
        """
        act_models = []

        for foot_traj_k, com_traj_k in zip(foot_traj, com_traj):
            act_models += self.walking_problem_one_step(
                timeStep, foot_traj_k, com_traj_k)
        running_action = act_models[:-1]
        terminal_action = act_models[-1]

        return crocoddyl.ShootingProblem(x0, running_action, terminal_action)

    def walking_problem_one_step(self, timeStep, foot_traj, com_traj):
        """
        Create action models for one full gait cycle (RR, RF, LR, LF swing).

        Args:
            timeStep (float): Time step duration.
            foot_traj (list): List of foot trajectories [rh, rf, lh, lf].
            com_traj (list): List of CoM trajectories for each swing phase.

        Returns:
            list: A sequence of action models for a full gait cycle.
        """
        # Get individual foot trajectories
        rh_traj, rf_traj, lh_traj, lf_traj = foot_traj

        act_models = []

        # ------------------ Double support ------------------ #
        initKnots = 2
        supportFootIds = [self.lfFootId, self.rfFootId,
                          self.lhFootId, self.rhFootId]
        if self.firstStep:
            act_models += [self.create_contact_action(
                timeStep, supportFootIds)for _ in range(initKnots)]

        # Walking phase
        # ------------------ RIGHT REAR SWING ------------------ #
        swingFootIds = [self.rhFootId]
        supportFootIds = [self.lfFootId, self.rfFootId, self.lhFootId]
        act_models += self.createFootstepModels(
            timeStep,
            com_traj[0],
            rh_traj,
            supportFootIds,
            swingFootIds
        )
        # ------------------ RIGHT FRONT SWING ------------------ #
        swingFootIds = [self.rfFootId]
        supportFootIds = [self.lfFootId, self.lhFootId, self.rhFootId]
        act_models += self.createFootstepModels(
            timeStep,
            com_traj[1],
            rf_traj,
            supportFootIds,
            swingFootIds
        )
        # ------------------ LEFT REAR SWING ------------------ #
        swingFootIds = [self.lhFootId]
        supportFootIds = [self.lfFootId, self.rfFootId, self.rhFootId]
        act_models += self.createFootstepModels(
            timeStep,
            com_traj[2],
            lh_traj,
            supportFootIds,
            swingFootIds
        )
        # ------------------ LEFT FRONT SWING ------------------ #
        swingFootIds = [self.lfFootId]
        supportFootIds = [self.rfFootId, self.lhFootId, self.rhFootId]
        act_models += self.createFootstepModels(
            timeStep,
            com_traj[3],
            lf_traj,
            supportFootIds,
            swingFootIds
        )

        # ------------------ Double support ------------------ #
        act_models += [self.create_contact_action(
            timeStep, supportFootIds)]

        return act_models

    def createFootstepModels(self, timeStep, com_trajectories, foot_trajectories, supportFootIds, swingFootIds):
        """
        Create Crocoddyl action models for a single footstep swing.

        Args:
            timeStep (float): Time step duration.
            com_trajectories (list): Desired CoM trajectory during swing.
            foot_trajectories (list): Desired swing foot trajectory.
            supportFootIds (list): IDs of feet in contact.
            swingFootIds (list): IDs of feet in swing.

        Returns:
            list: Action models for the swing phase and foot switch.
        """
        foot_swing_model = []
        swing_foot_tasks = []

        for com_task, foot_task in zip(com_trajectories, foot_trajectories):
            foot_swing_model += [
                self.create_swingfoot_action(
                    timeStep, supportFootIds, com_task, foot_task)
            ]

        # Final switching model to land the foot
        foot_switch_model = self.createFootSwitchModel(
            swingFootIds, swing_foot_tasks)

        return [*foot_swing_model, foot_switch_model]

    #################################################################################
    # ---------------------------------- Actions ---------------------------------- #
    #################################################################################

    def create_swingfoot_action(self, timeStep, supportFootIds, comTask, swingFootTask):
        """
        Create an action model for a swing phase of one leg.

        Args:
            timeStep (float): Time step duration.
            supportFootIds (list): IDs of feet in contact.
            comTask (np.ndarray): Desired center of mass (CoM) target.
            swingFootTask (tuple): (footId, SE(3)) tuple for swing foot.

        Returns:
            crocoddyl.ActionModel: Swing foot action model with CoM and swing foot cost.
        """
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 3 * len(supportFootIds)

        contactModel = self._create_contact_model(supportFootIds, nu)
        costModel = crocoddyl.CostModelSum(self.state, nu)
        self._add_contact_cost(costModel, supportFootIds, nu)
        self._add_com_cost(costModel, comTask, nu)
        self._add_swingfoot_cost(costModel, swingFootTask, nu)

        action_model = self._create_action_model(
            timeStep, contactModel, costModel, nu)

        return action_model

    def create_contact_action(self, timeStep, supportFootIds):
        """
        Create an action model for a full contact phase (no swing legs).

        Args:
            timeStep (float): Time step duration.
            supportFootIds (list): List of foot frame IDs in contact.

        Returns:
            crocoddyl.ActionModel: Action model with contact constraints and cost.
        """
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 3 * len(supportFootIds)

        # get contact and cost model
        contactModel = self._create_contact_model(supportFootIds, nu)
        costModel = crocoddyl.CostModelSum(self.state, nu)
        self._add_contact_cost(costModel, supportFootIds, nu)

        # create action model
        action_model = self._create_action_model(
            timeStep, contactModel, costModel, nu)

        return action_model

    #################################################################################
    # ---------------------------------- Models ----------------------------------- #
    #################################################################################

    def _create_action_model(self, timeStep, contactModel, costModel, nu):
        """
        Create an integrated action model using a specified dynamics model, control parametrization, and integrator.

        Args:
            timeStep (float): Time step duration.
            contactModel (crocoddyl.ContactModelMultiple): Contact model with supporting feet.
            costModel (crocoddyl.CostModelSum): Cost model including task objectives.
            nu (int): Dimension of control input.

        Returns:
            crocoddyl.IntegratedActionModelAbstract: Integrated action model (Euler or Runge-Kutta).
        """
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
        """
        Create a 3D multi-contact model for the given support feet.

        Args:
            supportFootIds (list): IDs of feet in contact.
            nu (int): Dimension of control input.

        Returns:
            crocoddyl.ContactModelMultiple: Contact model with all supporting feet added.
        """
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

    #################################################################################
    # ----------------------------------- Costs ----------------------------------- #
    #################################################################################

    def _add_swingfoot_cost(self, costModel, swingFootTask, nu):
        """
        Add foot tracking cost for swing foot trajectories.

        Args:
            costModel (crocoddyl.CostModelSum): Cost model
            swingFootTask (list of tuples): Each tuple contains:
                - ID (int): Frame ID of the swing foot.
                - SE3: Desired position and translation of the swing foot.
            nu (int): Dimension of the control input.
        """
        for i in swingFootTask:
            frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
                self.state, i[0], i[1].translation, nu
            )
            footTrack = crocoddyl.CostModelResidual(
                self.state, frameTranslationResidual
            )
            costModel.addCost(
                self.model.frames[i[0]].name +
                "_footTrack", footTrack, 1e6
            )

    def _add_contact_cost(self, costModel, supportFootIds, nu):
        """
        Add contact-related costs including:
            - Friction cone constraints
            - State regularization
            - Control regularization
            - Joint/state bounds

        Args:
            costModel (crocoddyl.CostModelSum): Cost model
            supportFootIds (list): IDs of feet in contact.
            nu (int): Dimension of the control input.
        """
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
                self.model.frames[i].name + "_frictionCone", frictionCone, 1e2
            )
        stateWeights = np.array(
            [10.0] * 3                       # base position (x, y, z)
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
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)

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
        """
        Add center-of-mass tracking cost.

        Args:
            costModel (crocoddyl.CostModelSum): Cost model
            com_d (np.array): Desired CoM position (3D).
            nu (int): Dimension of the control input.
        """
        com_residual = crocoddyl.ResidualModelCoMPosition(
            self.state, com_d, nu)
        comTrack = crocoddyl.CostModelResidual(self.state, com_residual)
        costModel.addCost("comTrack", comTrack, 1e6)

    #################################################################################
    # ----------------------------- Foot Switch action ---------------------------- #
    #################################################################################

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
                self.model.frames[i].name + "_contact", supportContactModel
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
                self.model.frames[i].name + "_frictionCone", frictionCone, 1e1
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
                    self.model.frames[i[0]].name +
                    "_footTrack", footTrack, 1e7
                )
                costModel.addCost(
                    self.model.frames[i[0]].name + "_impulseVel",
                    impulseFootVelCost,
                    1e6,
                )
        stateWeights = np.array(
            [0.0] * 3
            + [500.0] * 3
            + [0.01] * (self.model.nv - 6)
            + [10.0] * self.model.nv
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
                self.model.frames[i].name + "_impulse", supportContactModel
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
                    self.model.frames[i[0]].name +
                    "_footTrack", footTrack, 1e7
                )
        stateWeights = np.array(
            [1.0] * 6 + [10.0] * (self.model.nv - 6) + [10.0] * self.model.nv
        )
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.model.defaultState, 0
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
