import numpy as np
import random
np.seterr(divide='ignore', invalid='ignore')

class Bird:
    """
    Instantiations represent foraging oystercatchers
    """

    def __init__(self, unique_id, pos, model, dominance, specialization, mussel_foraging_efficiency,
                 cockle_foraging_efficiency, macoma_foraging_efficiency, worm_foraging_efficiency):
        self.model = model

        # individual parameters
        self.unique_id = unique_id
        self.dominance = dominance
        self.specialization = specialization
        self.pos = pos
        self.mussel_foraging_eff = mussel_foraging_efficiency
        self.cockle_foraging_eff = cockle_foraging_efficiency
        self.macoma_foraging_eff = macoma_foraging_efficiency
        self.worm_foraging_eff = worm_foraging_efficiency

        # initial foraging parameters
        self.start_foraging = None # number of steps after high tide todo: deze kan weg
        self.time_foraged = self.model.params["time_foraged_init"] #todo: welke initialisatie waarde ? let op tijdstap (dit is niet in uur)
        self.goal_reached = True

        # stomach, weight, energy goal
        self.stomach_content = self.model.params["init_stomach_content"] # g todo: waarmee initialiseren?
        self.weight = self.model.params["start_weight"] # reference weight on sept 1, g
        self.energy_goal = None #kJ
        self.energy_gain = 0 # energy already foraged kJ

        # stomach content and digestive rate
        max_digestive_rate = self.model.params["max_digestive_rate"] # g WtW / min KerstenVisser1996
        self.max_digestive_rate = max_digestive_rate * self.model.resolution_min # digestive rate per 10 minutes
        self.deposition_efficiency = self.model.params["deposition_efficiency"]
        self.BodyGramEnergyCont = self.model.params["BodyGramEnergyCont"]  # kJ/gram fat
        self.BodyGramEnergyReq = self.model.params["BodyGramEnergyReq"]  # kJ/gram (25% larger)
        self.minimum_weight = self.model.params["minimum_weight"]
        self.max_stomach_content = self.model.params["max_stomach_content"] # g WtW KerstenVisser1996

        # energy requirements
        self.thermo_a = self.model.params["thermo_a"]     # kerstenpiersma 1987 kJ/day
        self.thermo_b = self.model.params["thermo_b"]
        self.metabolic_a = self.model.params["metabolic_a"]
        self.metabolic_b = self.model.params["metabolic_b"]

        # interference parameters
        self.competitors_threshold = self.model.params["competitors_threshold"]
        self.a = self.model.params["a"]
        self.b = self.model.params["b"]
        self.attack_distance = self.model.params["attack_distance"]  # webtics, stillman 2002
        self.alpha = self.model.params["alpha"]

        # get some data
        self.weight_throughout_cycle = []
        self.stomach_content_list = []
        self.foraging_time_per_cycle = []
        self.start_foraging_list = []
        self.positions = []

    def step(self):
        """A model step. Move, then eat. """

        self.positions.append(self.pos)

        # determine energy goal at start of new tidal cycle and set gain to zero
        if self.model.new_tidal_cycle:

            # get some data todo: needed?
            self.weight_throughout_cycle.append(self.weight)

            # calculate goal and determine energy already gained
            self.energy_goal = self.energy_goal_coming_cycle(self.model.temperature,
                                                             self.model.total_number_steps_in_cycle)
            self.energy_gain = 0

            # calculate when to start foraging depending if goal was reached in prev cycle
            self.start_foraging = int(self.model.steps_to_low_tide - (self.time_foraged / 2))  # 0
            # if self.goal_reached:
            #     self.start_foraging = int(self.model.steps_to_low_tide - (self.time_foraged / 2))
            # else:
            #     self.start_foraging = int(self.model.steps_to_low_tide - (self.time_foraged / 2))# 0
            self.start_foraging_list.append(self.start_foraging)

            # keep track of time foraged in coming cycle
            self.time_foraged = 0

        # moving and foraging
        if self.model.time_in_cycle >= self.start_foraging and self.energy_gain < self.energy_goal:

            # check IR on patch
            if self.model.patch_types[self.pos] == "Bed":

                # interference intake reduction
                density_of_competitors = ((self.model.num_agents_on_patches[self.pos] - 1) * self.model.agg_factor_bed)\
                                         / self.model.available_areas[self.pos] #todo: density in model berekenen?
                relative_uptake = self.interference_stillman_float(density_of_competitors, self.dominance)

                # potential intake rate (kJ/s)
                potential_energy_intake_rate = self.model.mussel_potential_energy_intake * relative_uptake \
                                               / (self.model.resolution_min * 60) #todo: dit mss al in model berekenen?

                # potential IR is zero if available area of patch is zero
                if self.model.available_areas[self.pos] == 0:
                    potential_energy_intake_rate = 0

            elif self.model.patch_types[self.pos] == "Mudflat":

                # interference intake reduction
                density_of_competitors = ((self.model.num_agents_on_patches[self.pos] - 1) *
                                          self.model.agg_factor_mudflats) / self.model.available_areas[self.pos]
                relative_uptake = self.calculate_cockle_relative_intake(density_of_competitors, 1, 1)

                # energy intake current patch for cockle and macoma (kJ/s) todo: hier misschien functie van?
                energy_intake_cockle = self.model.energy_intake_cockle[self.pos] * relative_uptake
                energy_intake_mac = self.model.energy_intake_mac[self.pos]

                # get total energy intake per second per patch (kJ/s)
                potential_energy_intake_rate = (energy_intake_cockle + energy_intake_mac) / \
                                               (self.model.resolution_min * 60)

                # potential IR is zero if available area of patch is zero
                if self.model.available_areas[self.pos] == 0:
                    potential_energy_intake_rate = 0

            else:
                potential_energy_intake_rate = self.model.grassland_potential_energy_intake / (
                        self.model.resolution_min * 60)

            # if IR < threshold, move to other patch
            if potential_energy_intake_rate < self.model.leaving_threshold:
                self.move()

            # only forage if patch is available
            if self.model.available_areas[self.pos] > 0:

                # intake rate mussel bed
                if self.model.patch_types[self.pos] == "Bed":

                    # # calculate competitor density todo: dit is wss niet meer nodig door move()
                    # num_agents_on_patch = self.model.num_agents_on_patches[self.pos]
                    # density_of_competitors = (num_agents_on_patch - 1)/ self.model.available_areas[
                    #     self.pos]

                    # calculate intake
                    wtw_intake, energy_intake = self.consume_mussel_diet()

                    # update stomach content (add wet weight)
                    self.stomach_content += wtw_intake

                    # update energy gain (everything that is eaten)
                    self.energy_gain += energy_intake

                # intake rate mudflat
                elif self.model.patch_types[self.pos] == "Mudflat":

                    wtw_intake, energy_intake = self.consume_mudflats_diet() # todo: multiply with efficiency

                    # update stomach content (add wet weight)
                    self.stomach_content += wtw_intake

                    # update energy gain
                    self.energy_gain += energy_intake

            # intake rate grasslands (independent of available area)
            if self.model.patch_types[self.pos] == "Grassland":

                # intake rate becomes zero at low temperatures
                if (self.model.temperature < 0) | (self.model.day_night == 'N'):
                    wtw_intake, energy_intake = [0, 0]
                else:
                    wtw_intake, energy_intake = self.consume_grassland_diet()

                # update stomach content (add wet weight)
                self.stomach_content += wtw_intake

                # update energy gain
                self.energy_gain += energy_intake

        # digestion
        self.stomach_content -= min(self.max_digestive_rate, self.stomach_content)

        # at the end of the tidal cycle
        if self.model.time_in_cycle == self.model.total_number_steps_in_cycle - 1:

            # collect foraging time data
            self.foraging_time_per_cycle.append(self.time_foraged)

            # if energy goal not reached, start foraging immediately in next cycle
            if self.energy_gain < self.energy_goal:
                self.goal_reached = False
            else:
                self.goal_reached = True

            # energy consumption
            energy_consumed = self.energy_requirements_one_time_step(self.model.temperature) \
                              * self.model.total_number_steps_in_cycle

            # update weight
            energy_difference = self.energy_gain - energy_consumed
            if energy_difference < 0:
                self.weight += energy_difference / self.BodyGramEnergyCont
            elif energy_difference > 0:
                self.weight += energy_difference / self.BodyGramEnergyReq

            # apply death if weight becomes too low
            if self.weight < self.minimum_weight:
                self.model.schedule.remove(self)

    def move(self):
        """
        Agent moves if IR on patch is too low.

        Agent should find patch (depending on diet specialization) with IR > threshold.

        Depending on diet, if no patch is available, a bird stops foraging or moves to grassland.
        :return:
        """

        # calculate bird density on all patches
        all_patch_densities = self.model.num_agents_on_patches / self.model.available_areas # todo moet hier geen -1 op plek agent?

        # calculate intake rate on mudflats without interference (for both diet specialists)
        # todo: alleen patches boven ir berekenen? zo nee, dan is deze IR niet nodig
        # IR_mudflats_no_interf = self.model.mudflats_potential_energy_intake / (self.model.resolution_min * 60)

        ## todo: dit gedeelte is nu een beetje dubbel met het deel waarbij de agent op eigen patch kijkt wat de IR is

        # calculate relative intake rate for cockles on mudflats (based on densities)
        relative_cockle_intake = self.calculate_cockle_relative_intake(all_patch_densities *
                                                                       self.model.agg_factor_mudflats, 1, 1) #todo: haal onnodige variabelen weg

        # energy intake on all patches for cockle and macoma per second
        energy_intake_cockle_sec = (self.model.energy_intake_cockle * relative_cockle_intake) / \
                                   (self.model.resolution_min * 60)
        energy_intake_mac_sec = self.model.energy_intake_mac / (self.model.resolution_min * 60)

        # get total energy intake per second per patch todo: multiply with efficiency?
        total_patch_energy_intake = energy_intake_cockle_sec + energy_intake_mac_sec

        # patches with no available area have an intake of zero todo: to check
        total_patch_energy_intake[self.model.available_areas == 0] = 0

        # get indices of patches with IR that is large enough
        possible_positions = np.where(total_patch_energy_intake > self.model.leaving_threshold)[0] # todo: invalid value encountered in greater

        # if specialist is shellfish, also calculate IR on beds.
        if self.specialization == "shellfish":

            # density of competitors on mussel patches
            density_competitors_bed = all_patch_densities[: self.model.patch_max_bed_index + 1] \
                                      * self.model.agg_factor_bed# todo: mussel patches moeten dus bovenaan!

            # calculate relative IR for all mussel beds
            relative_mussel_intake = self.interference_stillman_array(density_competitors_bed, self.dominance)

            # calculate final IR on all mussel beds in kJ/s
            final_mussel_intake = relative_mussel_intake * self.model.mussel_potential_energy_intake / \
                                  (self.model.resolution_min * 60) # todo: multiply with efficiency?

            # patches with no available area have an intake of zero
            mask = (self.model.available_areas == 0)[: self.model.patch_max_bed_index + 1]
            final_mussel_intake[mask] = 0

            # get all possible indices and select mussel patches from that
            possible_positions_beds = np.where(final_mussel_intake > self.model.leaving_threshold)[0]
            possible_positions = np.concatenate([possible_positions, possible_positions_beds])

        # if there is no possible patch, stop foraging or move to grassland depending on diet
        if not len(possible_positions):
            if self.specialization == "worm":
                self.model.num_agents_on_patches[self.pos] -= 1
                self.pos = self.model.patch_index_grassland[0]
                self.model.num_agents_on_patches[self.pos] += 1

        # if there is a possible patch, choose a random new patch
        else:
            self.model.num_agents_on_patches[self.pos] -= 1
            self.pos = random.choice(possible_positions)
            self.model.num_agents_on_patches[self.pos] += 1

    def interference_stillman_float(self, density_competitors, local_dominance):
        """Helper method to calculate intake rate reduction as described in Stillman.

        This method takes the density_competitors as a float. (Use this function in case density competitors is a
        float since it is much faster than the array version for floats).
        :return:
        """

        # set density competitors to ha
        density_competitors = density_competitors * 10000

        # calculate relative intake rate
        if density_competitors > self.competitors_threshold:
            m = self.a + self.b * local_dominance
            relative_intake_rate = ((density_competitors + 1) / (self.competitors_threshold + 1)) ** -m
        else:
            relative_intake_rate = 1

        return relative_intake_rate

    def interference_stillman_array(self, density_competitors, local_dominance):
        """Helper method to calculate intake rate reduction as described in Stillman.

        Note that density_competitors should be given as np.array to ensure vector calculations work.
        :return:
        """

        # create array (in case density_competitors is float)
        density_competitors = np.array(density_competitors)

        # set density competitors to ha
        density_competitors = density_competitors * 10000

        # calculate relative intake rate
        m = self.a + self.b * local_dominance
        relative_intake_rate = np.where(density_competitors > self.competitors_threshold,
                                        ((density_competitors + 1) / (self.competitors_threshold + 1)) ** -m, 1)
        return relative_intake_rate

    def energy_requirements_one_time_step(self, T):
        """
        Calculate energy requirements for one time step.

        Included are thermoregulation and metabolic requirements. Note: weight gain is not included.

        Needs temperature for current time step

        Implementation uses same approach as in WEBTICS.
        :return: Energy for one time step in the model
        """

        # conversion from day to time step
        conversion = self.model.resolution_min / (24 * 60)

        # costs of thermoregulation for one time step # mss met lookup table? ipv elke keer berekenen?
        E_t = (self.thermo_a - T * self.thermo_b) * conversion # kJ/resolution_min

        # general required energy (for T > Tcrit) for one time step
        E_m  = (self.metabolic_a * self.weight ** self.metabolic_b) * conversion # kJ/conversion_min

        # return final energy requirement
        return max(E_t, E_m)

    def energy_goal_coming_cycle(self, mean_T, num_steps_tidal_cycle):
        """
        Method that calculates the energy goal of a bird for the coming tidal cycle.

        :param mean_T: Contains mean temperature for coming tidal cycle
        :return:
        """

        # determine energy for weight gain/loss
        weight_difference = self.model.reference_weight_birds - self.weight

        # check if bird should eat more/less for weight gain/loss
        if weight_difference < 0:
            weight_energy_requirement = self.BodyGramEnergyCont * weight_difference
        elif weight_difference > 0:
            weight_energy_requirement = self.BodyGramEnergyReq * weight_difference
        else:
            weight_energy_requirement = 0

        # calculate normal energy requirements
        energy_goal = self.energy_requirements_one_time_step(mean_T) * num_steps_tidal_cycle + weight_energy_requirement
        return energy_goal

    def consume_mussel_diet(self):
        """ Method that lets agent forage on mussel patch. Based on the energy goal and the stomach content
        the intake of an agent is evaluated.

        The patch depletion is also implemented.

        Returns the wet weight consumed (g).
        """

        # calculate competitor density
        num_agents_on_patch = self.model.num_agents_on_patches[self.pos]
        density_of_competitors = ((num_agents_on_patch - 1) / self.model.available_areas[self.pos]) \
                                 * self.model.agg_factor_bed

        # # interference intake reduction
        relative_uptake = self.interference_stillman_float(density_of_competitors, self.dominance)

        # wet intake rate (intake rate including interference and foraging efficiency)
        wtw_intake = self.model.mussel_potential_wtw_intake * relative_uptake * self.mussel_foraging_eff

        # calculate possible intake based on stomach left and digestive rate
        possible_wtw_intake = self.calculate_possible_intake() # g / 10 minutes

        # intake is minimum of possible intake and intake achievable on patch todo: zelfde als bij mudflats maken
        intake_wtw = min(wtw_intake, possible_wtw_intake)  # WtW intake in g

        # calculate actual energy intake
        energy_intake = intake_wtw * self.model.FractionTakenUp * self.model.RatioAFDWtoWet \
                        * self.model.AFDWenergyContent

        # check if energy gain does not exceed goal, if so, adapt intake #todo: functie van maken?
        if self.energy_gain + energy_intake > self.energy_goal:

            # calculate surplus
            surplus = self.energy_gain + energy_intake - self.energy_goal

            # fraction of this time step needed to accomplish goal
            fraction_needed = 1 - (surplus / energy_intake)

            # multiply all intakes with fraction needed
            intake_wtw *= fraction_needed
            energy_intake *= fraction_needed

            # update foraging time
            self.time_foraged += fraction_needed
        else:
            self.time_foraged += 1
        return intake_wtw, energy_intake

    def consume_mudflats_diet(self):
        """ Method that lets agent forage on mudflat (currently only cockles taken into account).

        In this method the depletion of prey on a patch is also implemented.

        :return: The amount of wet weight foraged is returned (in g / time step). The energy
        is also returned in kJ/time step.
        """

        # for cockles, calculate uptake reduction
        bird_density = ((self.model.num_agents_on_patches[self.pos] - 1) / self.model.available_areas[self.pos]) \
                       * self.model.agg_factor_mudflats

        # individual relative intake (fraction)
        relative_intake = self.calculate_cockle_relative_intake(bird_density, self.attack_distance, self.alpha)

        # get the capture rate of all prey on mudflat (different cockle sizes)
        total_captured_kok1, total_captured_kok2, total_captured_kokmj, total_captured_mac \
            = self.model.capture_rates_mudflats

        # only get captured prey from current patch including interference
        total_captured_kok1, total_captured_kok2, total_captured_kokmj, total_captured_mac \
            = total_captured_kok1[self.pos] * relative_intake * self.cockle_foraging_eff, \
              total_captured_kok2[self.pos] * relative_intake * self.cockle_foraging_eff, \
              total_captured_kokmj[self.pos] * relative_intake * self.cockle_foraging_eff, \
              total_captured_mac[self.pos] * self.macoma_foraging_eff

        # wet weight intake, note that we should use capture rate including interference (and not global wtw intake)
        # patch_wtw_intake = self.model.mudflats_potential_wtw_intake[self.pos] * relative_intake #
        # wet weight intake rate (g/s)
        patch_wtw_intake = total_captured_kok1 * self.model.cockle_wet_weight[self.pos][0] \
                             + total_captured_kok2 * self.model.cockle_wet_weight[self.pos][1]\
                             + total_captured_kokmj * self.model.cockle_wet_weight[self.pos][2]\
                             + total_captured_mac * self.model.macoma_wet_weight[self.pos]
        patch_wtw_intake *= (1 - self.model.LeftOverShellfish)

        # calculate possible intake based on stomach left and digestive rate
        possible_wtw_intake = self.calculate_possible_intake()  # g / time step

        # intake is minimum of possible intake and intake achievable on patch
        intake_wtw = min(patch_wtw_intake, possible_wtw_intake)   # WtW intake in g

        # compare final intake to original patch intake
        if patch_wtw_intake > 0:
            fraction_possible_final_intake = intake_wtw / patch_wtw_intake
        else:
            fraction_possible_final_intake = 0

        # calculate final number of prey eaten
        final_captured_kok1 = total_captured_kok1 * fraction_possible_final_intake
        final_captured_kok2 = total_captured_kok2 * fraction_possible_final_intake
        final_captured_kokmj = total_captured_kokmj * fraction_possible_final_intake
        final_captured_mac = total_captured_mac * fraction_possible_final_intake

        # calculate energy intake
        energy_intake = intake_wtw * self.model.FractionTakenUp * self.model.RatioAFDWtoWet \
                                    * self.model.AFDWenergyContent

        # check if energy gain does not exceed goal, if so, adapt intake # todo:" gebeurd dit niet al in fraction_possible_intake?
        if self.energy_gain + energy_intake > self.energy_goal:
            # calculate surplus
            surplus = self.energy_gain + energy_intake - self.energy_goal

            # fraction of this time step needed to accomplish goal
            fraction_needed = 1 - (surplus / energy_intake)

            # multiply all intakes with fraction needed todo: check dit nog een keer
            intake_wtw *= fraction_needed
            energy_intake *= fraction_needed
            final_captured_kok1 *= fraction_needed
            final_captured_kok2 *= fraction_needed
            final_captured_kokmj *= fraction_needed
            final_captured_mac *= fraction_needed

            # update foraging time
            self.time_foraged += fraction_needed
        else:
            self.time_foraged += 1

        # deplete prey (use actual area of patch)
        self.model.cockle_densities[self.pos][0] -= final_captured_kok1 / self.model.patch_areas[self.pos]
        self.model.cockle_densities[self.pos][1] -= final_captured_kok2 / self.model.patch_areas[self.pos]
        self.model.cockle_densities[self.pos][2] -= final_captured_kokmj / self.model.patch_areas[self.pos]
        self.model.macoma_density[self.pos] -= final_captured_mac / self.model.patch_areas[self.pos]
        return intake_wtw, energy_intake

    def consume_grassland_diet(self):

        potential_wtw_intake, potential_energy_intake = \
            self.model.grassland_potential_wtw_intake * self.worm_foraging_eff, \
            self.model.grassland_potential_energy_intake * self.worm_foraging_eff

        # calculate possible wtw intake based on stomach left and digestive rate
        possible_wtw_intake = self.calculate_possible_intake()  # g / time step

        # intake is minimum of possible intake and intake achievable on patch
        final_intake_wtw = min(potential_wtw_intake, possible_wtw_intake) * self.model.FractionTakenUp # WtW intake in g

        # calculate energy intake, multiply with fraction of possible intake divided by max intake
        energy_intake = potential_energy_intake * final_intake_wtw / potential_wtw_intake  # kJ

        # check if energy gain does not exceed goal, if so, adapt intake #todo: dit in functie? nu driedubbel
        if self.energy_gain + energy_intake > self.energy_goal:

            # calculate surplus
            surplus = self.energy_gain + energy_intake - self.energy_goal

            # fraction of this time step needed to accomplish goal
            fraction_needed = 1 - (surplus / energy_intake)

            # multiply all intakes with fraction needed
            final_intake_wtw *= fraction_needed
            energy_intake *= fraction_needed

            # update foraging time
            self.time_foraged += fraction_needed
        else:
            self.time_foraged += 1
        return final_intake_wtw, energy_intake

    def calculate_possible_intake(self):
        """ Method calculated the intake rate a bird can have (which depends on how full its stomach is and also
        the digestive rate)
        """

        # check stomach space left
        stomach_left = self.max_stomach_content - self.stomach_content  # g

        # calculate possible intake based on stomach left and digestive rate
        possible_wtw_intake = self.max_digestive_rate + stomach_left  # g / time step
        return possible_wtw_intake

    def calculate_cockle_relative_intake(self, bird_density, attack_distance, alpha):
        """ Method that calculates the uptake reduction for the cockle intake rate due to the
        presence of competitors
        """

        exponent = -np.pi * bird_density * (self.attack_distance ** 2) * self.alpha #todo: moet hier density -1?
        relative_intake = np.exp(exponent)
        return relative_intake
