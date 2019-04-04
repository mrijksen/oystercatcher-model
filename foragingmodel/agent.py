
import numpy as np

class Bird:
    """
    Instantiations represent foraging oystercatchers
    """

    # parameters that are same for every bird #
    # max stomach content
    max_stomach_content = 80 # g WtW

    def __init__(self, unique_id, pos, model, dominance, energy=None):
        self.unique_id = unique_id
        self.model = model
        self.dominance = dominance
        self.energy = energy
        self.pos = pos

        # variable indicating "when" agent starts foraging (time steps after start tidal cycle) todo: zet in mooie units
        self.start_foraging = 3 * 60 / model.resolution_min # todo: let op dat je 0 wel meerekent! Nu na 3 uur (3.25u voor laagwater)

        # stomach content
        self.stomach_content = 0 # todo: waarmee initialiseren?

        # weight todo: what initial weight?
        self.weight = 500 # gram

        # todo: add stomach etc.

    def step(self): # todo:
        """A model step. Move, then eat. """
        # print("Agent id:", self.unique_id, "pos:", self.pos)
        # determine whether to move

            # in case we move, choose other patch

        # if new tidal cycle, calculate new energy goal

        # start foraging:
        if self.model.time_in_cycle == self.start_foraging:
            print("start foraging now!")


        # num of other agents and calculate local dominance
        num_agents_on_patch, local_dominance = self.calculate_local_dominance(self.model)

        # calculate competitor density
        density_of_competitors = num_agents_on_patch / self.model.patch_areas[self.pos] #todo: in stillman is dit in ha


        # capture and intake rate including interference todo: is dit beide nodig?
        capture_rate, intake_rate = self.intake_rate_mussel(self.model.prey[self.pos], self.model.init_mussel_weight,
                                              density_of_competitors, local_dominance) #todo: make mussel weight change
        # print("Intake rate: ", intake_rate)
        # print("capture rate", capture_rate)

        # get total intake over time step
        total_captured_num_mussels = capture_rate * self.model.resolution_min * 60

        intake_dry_weight = intake_rate * self.model.resolution_min * 60 # gebruiken voor energy storage

        # apply death #todo: should this be above eat?

        # deplete prey on patch
        self.model.prey[self.pos] -= total_captured_num_mussels/ self.model.patch_areas[self.pos]

        # print("Dominance: {}, L: {}, Intake Rate {}".format(self.dominance, self.))


        # update stomach and energy reserves


    def intake_rate_mussel(self, mussel_density, prey_weight, density_competitors, local_dominance):
        """Calculate intake rate for mussel patch on Wadden Sea.

        Functional response is derived from WEBTICS.

        Interference is derived from Stillman et al. (2000).

        Final intake rate is in mg.
        """

        # parameters
        attack_rate = 0.00057 # mosselA in stillman
        max_intake_rate = self.maximal_intake_rate(prey_weight)

        # interference intake reduction
        interference = self.interference_stillman(density_competitors, local_dominance)

        # calculate capture rate and include interference
        capture_rate = self.functional_response_mussel(attack_rate, mussel_density, prey_weight, max_intake_rate)
        final_capture_rate = capture_rate * interference

        # calculate actual dry weight intake
        dry_weight_intake_rate = capture_rate * prey_weight

        # final intake rate included interference
        final_intake_rate = dry_weight_intake_rate * interference
        # print("final IR:", final_intake_rate)
        return final_capture_rate, final_intake_rate

    @staticmethod
    def functional_response_mussel(attack_rate, mussel_density, prey_weight, max_intake_rate):
        """
        Functional response as described in WEBTICS. They converted
        the intake of stillman to a capture rate.

        :param attack_rate:
        :param max_intake_rate:
        :param prey_weight: average dry mass weight of mussels (no size classes)
        :return: capture rate in # prey/ s
        """

        # calculate handling time and capture rate
        handling_time = prey_weight / max_intake_rate
        capture_rate = attack_rate * mussel_density / (1 + attack_rate * handling_time * mussel_density)
        # print("capture_rate:", capture_rate)
        return capture_rate

    @staticmethod
    def maximal_intake_rate(prey_weight):
        """Calculate maximal intake rate as described in WEBTICS (page 62)

        :return max intake rate in mg
        """

        # parameters
        mussel_intake_rate_A = 0.092  # parameters for max intake rate (plateau)
        mussel_intake_rate_B = 0.506

        # calculate plateau/max intake rate
        max_intake_rate = mussel_intake_rate_A * prey_weight ** mussel_intake_rate_B
        # print("max IR:", max_intake_rate)
        return max_intake_rate

    @staticmethod #todo: moet dit in staticfunction?
    def interference_stillman(density_competitors, local_dominance):
        """Helper method to calculate intake rate reduction as described in Stillman.
        :return:
        """

        # todo: what to do with the number of days since september 1? exclude it maybe? as in 1996 paper?

        # parameters
        competitors_threshold = 0 # density of competitors above which interference occurs
        a = 0.437 # parameters for stabbers as described by Stillman 1996
        b = -0.00721 #todo: check 587 for threshold

        # calculate relative intake rate
        if density_competitors > competitors_threshold:
            m = a + b * local_dominance
            relative_intake_rate = ((density_competitors + 1) / (competitors_threshold + 1)) ** -m
        else:
            relative_intake_rate = 1
        return relative_intake_rate

    def calculate_local_dominance(self, model):
        """
        Method that calculates local dominance (# of encounters won) for patch agent is currently on

        Returns number of other agents on same patch and number of encounters won (L)
        """

        # find dominance of all agents on same patch (excluding self)
        dominance_agents_same_patch = [agent.dominance for agent in model.agents_on_patches[self.pos]
                                if agent.unique_id != self.unique_id]

        # calculate number of encounters won
        number_of_encounters = len(dominance_agents_same_patch)

        if number_of_encounters == 0:
            L = 0 #todo: klopt dit? ja want die andere term wordt toch 1 (van interference)
        else:
            agents_with_lower_dominance = [item for item in dominance_agents_same_patch if item < self.dominance] #todo: smaller then or equal?
            L = (len(agents_with_lower_dominance) / number_of_encounters) * 100
        return len(dominance_agents_same_patch), L

    def calculate_energy_requirements(self, T_list):
        """
        Calculate energy requirements for coming tidal cycle.

        Included are thermoregulation and metabolic requirements. Note: weight gain is not included.

        Needs a list with temperature for coming time steps

        Implementation uses same approach as in WEBTICS.
        :return:
        """

        # parameters
        thermo_a = 904      # kerstenpiersma 1987
        thermo_b = 30.3
        metabolic_a = 0.061 # zwartsenskerstenetal1996
        metabolic_b = 1.489

        for T in T_list: # todo: mss lookup table gebruiken ipv berekening?

            # costs of thermoregulation #todo: dit is per dag, moet per tijdstap a resolution_min
            E_t = thermo_a - T * thermo_b

            # general required energy (for T > Tcrit)
            E_m = metabolic_a * self.weight ** metabolic_b

            # total energy requirement


















