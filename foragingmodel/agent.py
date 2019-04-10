class Bird:
    """
    Instantiations represent foraging oystercatchers
    """

    def __init__(self, unique_id, pos, model, dominance):

        # standard params
        self.unique_id = unique_id
        self.model = model

        # this should be read from data file
        self.dominance = dominance
        self.pos = pos

        # variable indicating "when" agent starts foraging (time steps after start tidal cycle) todo: zet in mooie units
        self.start_foraging = 3 * 60 / model.resolution_min # todo: let op dat je 0 wel meerekent! Nu na 3 uur (3.25u voor laagwater)
        # self.start_foraging = 0

        # stomach, weight, energy goal
        self.stomach_content = 0 # todo: waarmee initialiseren?
        self.weight = 300
        self.energy_goal = None
        self.energy_gain = 0 # energy already foraged

        # stomach content en digestive rate todo: put in parameter file
        self.max_stomach_content = 80  # g WtW
        max_digestive_rate = 0.263 # WtW / min KerstenVisser1996
        self.max_digestive_rate = max_digestive_rate * 10 # digestive rate per 10 minutes
        self.deposition_efficiency = 0.75  # WEBTICS page 57
        self.BodyGramEnergyCont = 34.295  # kJ/gram fat
        self.BodyGramEnergyReq = 45.72666666  # kJ/gram (25% larger)
        self.minimum_weight = 30 # todo
        self.max_stomach_content = 80 # g WtW KerstenVisser1996
        self.fraction_taken_up = 0.85 # Speakman1987, KerstenVisser1996, KerstenPiersma1987, ZwartsBlomert1996 #todo: moet er nog in

        # get some data
        self.weight_throughout_cycle = []
        self.stomach_content_list = []

    def step(self): # todo:
        """A model step. Move, then eat. """
        print("Agent id:", self.unique_id, "pos:", self.pos, "weight:", self.weight)

        # get some data
        self.stomach_content_list.append(self.stomach_content)
        self.weight_throughout_cycle.append(self.weight)

        # determine energy goal at start of new tidal cycle and set gain to zero
        if self.model.time_in_cycle == 0:
            self.energy_goal = self.energy_goal_coming_cycle(self.model.temperature) #todo: what temperature?
            self.energy_gain = 0

        # foraging
        if self.model.time_in_cycle >= self.start_foraging and self.energy_gain < self.energy_goal: #todo: move if patch not available

            # num of other agents and calculate local dominance todo: for some patches this is not needed
            num_agents_on_patch, local_dominance = self.calculate_local_dominance(self.model)  # todo: num_agents moet geupdate worden

            # calculate competitor density
            density_of_competitors = num_agents_on_patch / self.model.patch_areas[self.pos]  # todo: in stillman is dit in ha, dit kan ook in functie calculate local dom

            # todo: check patch type and calculate intake on that patch based on available prey and competitors

            wtw_intake = self.consume_mussel_diet(density_of_competitors, local_dominance) #todo: depends on patch

            # update stomach content (add wet weight)
            self.stomach_content += wtw_intake

            # update energy gain (everything that is eaten)
            self.energy_gain += wtw_intake * self.model.RatioAFDWtoWet * self.model.AFDWenergyContent

        # only digested food is assimilated
        energy_assimilated = (min(self.max_digestive_rate,
                             self.stomach_content)) * self.model.RatioAFDWtoWet * self.model.AFDWenergyContent  # todo: fractiontakenup?

        # digestion
        self.stomach_content -= min(self.max_digestive_rate, self.stomach_content) # TODO DIT MOET NAAR TIJDSTAP OMGEZET

        # energy consumption
        energy_consumed = self.energy_requirements_one_time_step(self.model.temperature)

        # update weight todo: do this every time step or only at end of tidal cycle?
        energy_difference = energy_assimilated - energy_consumed
        self.weight += energy_difference / self.BodyGramEnergyCont
        print("Energy assimilated:", energy_assimilated, "Energy Consumed", energy_consumed)
        # print("Energy intake", wtw_intake * self.model.RatioAFDWtoWet * self.model.AFDWenergyContent)

        # apply death if weight becomes too low
        if self.weight < self.minimum_weight: #todo: this should be something else maybe?
            self.model.schedule.remove(self)
        # print("Weight;", self.weight, "Egain:", self.energy_gain)

    def intake_rate_mussel(self, mussel_density, prey_dry_weight, prey_wet_weight, density_competitors, local_dominance):
        """Calculate intake rate for mussel patch on Wadden Sea.

        Functional response is derived from WEBTICS.

        Interference is derived from Stillman et al. (2000).

        Final intake rate is in mg/s.
        """

        # parameters
        attack_rate = 0.00057 # mosselA in stillman
        max_intake_rate = self.maximal_intake_rate(prey_dry_weight) #todo haal dry weight weg

        # interference intake reduction
        interference = self.interference_stillman(density_competitors, local_dominance)

        # calculate capture rate and include interference
        capture_rate = self.functional_response_mussel(attack_rate, mussel_density, prey_dry_weight, max_intake_rate)
        final_capture_rate = capture_rate * interference

        # calculate actual dry weight intake
        dry_weight_intake_rate = capture_rate * prey_dry_weight #todo: dit kan er misschien wel uit

        # calculate actual wet weight intake
        wet_weight_intake_rate = capture_rate * prey_wet_weight #todo moet dit erin?

        # final intake rate included interference
        final_intake_rate = dry_weight_intake_rate * interference
        # print("final IR:", final_intake_rate)
        return final_capture_rate, final_intake_rate, wet_weight_intake_rate

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
        return capture_rate

    @staticmethod
    def maximal_intake_rate(prey_weight):
        """Calculate maximal intake rate as described in WEBTICS (page 62)

        :return max intake rate in mg
        """

        # parameters todo: in parameter file
        mussel_intake_rate_A = 0.092  # parameters for max intake rate (plateau)
        mussel_intake_rate_B = 0.506

        # calculate plateau/max intake rate
        max_intake_rate = mussel_intake_rate_A * prey_weight ** mussel_intake_rate_B
        return max_intake_rate

    @staticmethod #todo: moet dit in staticfunction?
    def interference_stillman(density_competitors, local_dominance):
        """Helper method to calculate intake rate reduction as described in Stillman.
        :return:
        """

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
            L = 0
        else:
            agents_with_lower_dominance = [item for item in dominance_agents_same_patch if item < self.dominance] #todo: smaller then or equal?
            L = (len(agents_with_lower_dominance) / number_of_encounters) * 100
        return len(dominance_agents_same_patch), L

    def energy_requirements_one_time_step(self, T):
        """
        Calculate energy requirements for one time step.

        Included are thermoregulation and metabolic requirements. Note: weight gain is not included.

        Needs temperature for current time step

        Implementation uses same approach as in WEBTICS.
        :return:
        """

        # conversion from day to time step
        conversion = self.model.resolution_min / (24 * 60)

        # parameters
        thermo_a = 904     # kerstenpiersma 1987 kJ/day
        thermo_b = 30.3
        metabolic_a = 0.061 # zwartsenskerstenetal1996 kJ/day
        metabolic_b = 1.489

        # costs of thermoregulation for one time step # mss met lookup table? ipv elke keer berekenen?
        E_t = (thermo_a - T * thermo_b) * conversion # kJ/resolution_min

        # general required energy (for T > Tcrit) for one time step
        E_m  = (metabolic_a * self.weight ** metabolic_b) * conversion # kJ/conversion_min

        # return final energy requirement
        return max(E_t, E_m)

    def energy_goal_coming_cycle(self, mean_T): #todo: over 1 or 2 tidal cycles? 2 lijkt logisch, en handig als je dag en  nacht meeneemt
        """
        Method that calculates the energy goal of a bird for the coming tidal cycle.

        :param mean_T: Contains mean temperature for coming or previous tidal cycle TODO: prev or coming?
        :return:
        """

        # determine energy for weight gain/loss
        weight_difference = self.model.reference_weight_birds - self.weight
        print("weight difference", weight_difference)

        # todo: this weight energy does not take into account time step size. Is this a bad thing?
        # todo: I don't think so, Egoal will be high but then they will eat as much as possible (as should)
        # todo: Turn weight data into weight per tidal cycle.
        if weight_difference != 0:
            weight_energy_requirement = self.BodyGramEnergyReq * weight_difference
        else:
            weight_energy_requirement = 0
        energy_goal = weight_energy_requirement # todo: unnessesary variable just for clarity

        # calculate normal energy requirements
        energy_goal += self.energy_requirements_one_time_step(mean_T) * self.model.steps_per_tidal_cycle

        return energy_goal

    def consume_mussel_diet(self, density_of_competitors, local_dominance):
        """ Method that lets agent forage on mussel patch. Based on the energy goal and the stomach content
        the intake of an agent is evaluated.

        Returns the energy (kJ) assimilated and the wet weight consumed (g).
        """
        #todo: ze doen nu hele tijdstap hetzelfde, moeten iets toevoegen zodat ze deel van tijdstap kunnen forageren
        # check if energy goal is met
        print("energy gain", self.energy_gain)
        print("energy goal", self.energy_goal)

        # # eat if energy goal is not reached
        # if self.energy_gain < self.energy_goal:
        #
        #     # num of other agents and calculate local dominance
        #     num_agents_on_patch, local_dominance = self.calculate_local_dominance(self.model) #todo: num_agents moet geupdate worden
        #
        #     # calculate competitor density
        #     density_of_competitors = num_agents_on_patch / self.model.patch_areas[self.pos]  # todo: in stillman is dit in ha

        # capture and intake rate including interference todo: voeg andere IRs toe afhankelijk van dieet
        patch_capture_rate, patch_intake_rate_dry_weight, patch_intake_rate_wet_weight = \
            self.intake_rate_mussel(self.model.prey[self.pos], self.model.init_mussel_dry_weight,
                                    self.model.init_mussel_wet_weight,
                                    density_of_competitors, local_dominance)  # todo: make mussel weight change, this structure has to change if data changes

        # get total capture rate/IRs in one time step todo: dit misschien in functie (intake_rate_mussel) zetten?
        conversion_s_to_timestep = self.model.resolution_min * 60
        total_patch_intake_wet_weight = patch_intake_rate_wet_weight * conversion_s_to_timestep #todo: moet dit allemaal berekent worden?

        # check stomach space left
        stomach_left = self.max_stomach_content - self.stomach_content

        # calculate possible intake based on stomach left and digestive rate
        possible_wtw_intake = self.max_digestive_rate + stomach_left

        # intake is minimum of possible intake and intake achievable on patch
        intake_wtw = min(total_patch_intake_wet_weight, possible_wtw_intake) # WtW intake

        # num prey captured
        num_prey_captured = int(intake_wtw / self.model.init_mussel_wet_weight) #todo: hier de leftover fraction?

        # update patch density
        self.model.prey[self.pos] -= num_prey_captured / self.model.patch_areas[self.pos]
        return intake_wtw