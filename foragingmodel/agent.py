import numpy as np
import random
np.seterr(divide='ignore')

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
        self.start_foraging = None # number of steps after high tide
        self.time_foraged = 6 #todo: welke initialisatie waarde ? let op tijdstap (dit is niet in uur)

        # stomach, weight, energy goal
        self.stomach_content = 0 # g todo: waarmee initialiseren?
        self.weight = 548.9 # reference weight on sept 1, g
        self.energy_goal = None #kJ
        self.energy_gain = 0 # energy already foraged kJ

        # stomach content en digestive rate todo: put in parameter file
        max_digestive_rate = 0.263 # g WtW / min KerstenVisser1996
        self.max_digestive_rate = max_digestive_rate * self.model.resolution_min # digestive rate per 10 minutes
        self.deposition_efficiency = 0.75  # WEBTICS page 57
        self.BodyGramEnergyCont = 34.295  # kJ/gram fat todo: moet dit niet een soort rate zijn? Nu kan het oneindig snel van en naar gewicht gaan
        self.BodyGramEnergyReq = 45.72666666  # kJ/gram (25% larger)
        self.minimum_weight = 450 # todo: dit is een algemene parameter
        self.max_stomach_content = 80 # g WtW KerstenVisser1996

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

            # get some data
            # self.stomach_content_list.append(self.stomach_content)
            self.weight_throughout_cycle.append(self.weight)

            # calculate goal and determine energy already gained
            self.energy_goal = self.energy_goal_coming_cycle(self.model.temperature,
                                                             self.model.total_number_steps_in_cycle)
            self.energy_gain = 0

            # calculate when to start foraging
            self.start_foraging = int(self.model.steps_to_low_tide - (self.time_foraged / 2))
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

            elif self.model.patch_types[self.pos] == "Grassland":
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

                # intake rate grasslands
                elif self.model.patch_types[self.pos] == "Grassland":

                    # intake rate becomes zero at low temperatures # todo: hier nacht invoegen
                    if (self.model.temperature < 0) | (self.model.day_night == 'N'):
                        wtw_intake, energy_intake = [0, 0]
                    else:
                        wtw_intake, energy_intake = self.consume_grassland_diet() #todo: multiply with efficiency

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
                self.time_foraged = 1000 # todo: not neat to do like this maybe? FIX

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
                self.pos = self.model.patch_index_grassland
                self.model.num_agents_on_patches[self.pos] += 1
            else:
                # todo: move to roost?
                pass

        # if there is a possible patch, choose a random new patch
        else:

            self.model.num_agents_on_patches[self.pos] -= 1
            self.pos = random.choice(possible_positions)
            self.model.num_agents_on_patches[self.pos] += 1


    @staticmethod
    def interference_stillman_float(density_competitors, local_dominance):
        """Helper method to calculate intake rate reduction as described in Stillman.

        This method takes the density_competitors as a float. (Use this function in case density competitors is a
        float since it is much faster than the array version for floats).
        :return:
        """

        # parameters
        competitors_threshold = 158 # density of competitors above which interference occurs ha-1 todo: welke nemen?
        a = 0.437 # parameters for stabbers as described by Stillman 1996
        b = -0.00721 #todo: check 587 for threshold

        # set density competitors to ha
        density_competitors = density_competitors * 10000

        # calculate relative intake rate
        if density_competitors > competitors_threshold:
            m = a + b * local_dominance
            relative_intake_rate = ((density_competitors + 1) / (competitors_threshold + 1)) ** -m
        else:
            relative_intake_rate = 1

        return relative_intake_rate

    @staticmethod
    def interference_stillman_array(density_competitors, local_dominance):
        """Helper method to calculate intake rate reduction as described in Stillman.

        Note that density_competitors should be given as np.array to ensure vector calculations work.
        :return:
        """

        # create array (in case density_competitors is float)
        density_competitors = np.array(density_competitors)

        # parameters
        competitors_threshold = 158  # density of competitors above which interference occurs ha-1 todo: welke nemen?
        a = 0.437  # parameters for stabbers as described by Stillman 1996
        b = -0.00721  # todo: check 587 for threshold

        # set density competitors to ha
        density_competitors = density_competitors * 10000

        # calculate relative intake rate
        m = a + b * local_dominance
        relative_intake_rate = np.where(density_competitors > competitors_threshold,
                                        ((density_competitors + 1) / (competitors_threshold + 1)) ** -m, 1)
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

        # parameters
        attack_distance = 2.0  # webtics, stillman 2002
        alpha = 0.4  # fitted parameter by webtics

        # individual relative intake (fraction)
        relative_intake = self.calculate_cockle_relative_intake(bird_density, attack_distance, alpha)

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

    @staticmethod #todo: haal onnodige parameters weg
    def calculate_cockle_relative_intake(bird_density, attack_distance, alpha):
        """ Method that calculates the uptake reduction for the cockle intake rate due to the
        presence of competitors
        """

        # parameters
        attack_distance = 2.0  # webtics, stillman 2002
        alpha = 0.4  # fitted parameter by webtics

        exponent = -np.pi * bird_density * (attack_distance ** 2) * alpha #todo: moet hier density -1?
        relative_intake = np.exp(exponent)
        return relative_intake

    # def consume_mussel_diet(self, density_of_competitors, local_dominance):
    #     """ Method that lets agent forage on mussel patch. Based on the energy goal and the stomach content
    #     the intake of an agent is evaluated.
    #
    #     The patch depletion is also implemented.
    #
    #     Returns the wet weight consumed (g).
    #     """
    #
    #     # capture and intake rate including interference
    #     patch_capture_rate = self.capture_rate_mussel(self.model.mussel_afdw, density_of_competitors,
    #                                                  local_dominance) # todo: kies tussen self. of staticfunction
    #
    #     # wet intake rate
    #     patch_wet_intake = patch_capture_rate * self.model.mussel_wet_weight # g WtW/s
    #     patch_wet_intake *= (1 - self.model.LeftOverShellfish)
    #
    #     # get total capture rate/IRs in one time step
    #     # todo: kan dit buiten functie? dan gebruiken we het voor alle patch types
    #     conversion_s_to_timestep = self.model.resolution_min * 60
    #     total_patch_intake_wet_weight = patch_wet_intake * conversion_s_to_timestep # g/time step
    #
    #     # calculate possible intake based on stomach left and digestive rate
    #     possible_wtw_intake = self.calculate_possible_intake() # g / 10 minutes
    #
    #     # intake is minimum of possible intake and intake achievable on patch
    #     intake_wtw = min(total_patch_intake_wet_weight, possible_wtw_intake)  # WtW intake in g
    #
    #     # calculate energy intake
    #     energy_intake = intake_wtw * self.model.FractionTakenUp * self.model.RatioAFDWtoWet * self.model.AFDWenergyContent
    #
    #     # check if energy gain does not exceed goal, if so, adapt intake #todo: functie van maken?
    #     if self.energy_gain + energy_intake > self.energy_goal:
    #
    #         # calculate surplus
    #         surplus = self.energy_gain + energy_intake - self.energy_goal
    #
    #         # fraction of this time step needed to accomplish goal
    #         fraction_needed = 1 - (surplus / energy_intake)
    #
    #         # multiply all intakes with fraction needed
    #         intake_wtw *= fraction_needed
    #         energy_intake *= fraction_needed
    #
    #         # update foraging time
    #         self.time_foraged += fraction_needed
    #     else:
    #         self.time_foraged += 1
    #     return intake_wtw, energy_intake

    # def capture_rate_mussel(self, prey_dry_weight, density_competitors, local_dominance):
    #     """Calculate capture rate for mussel patch on Wadden Sea.
    #
    #     Functional response is derived from WEBTICS.
    #
    #     Interference is derived from Stillman et al. (2000).
    #
    #     Weight of prey should be given in g.
    #
    #     Final capture rate is in #/s.
    #     """
    #
    #     # todo maak hier met functional response 1 functie van
    #
    #     # parameters
    #     attack_rate = 0.00057 # mosselA in stillman
    #     max_intake_rate = self.maximal_intake_rate(prey_dry_weight) #todo haal dry weight weg
    #
    #     # interference intake reduction
    #     interference = self.interference_stillman(density_competitors, local_dominance)
    #
    #     # calculate capture rate and include interference
    #     capture_rate = self.functional_response_mussel(attack_rate, self.model.mussel_density, prey_dry_weight, max_intake_rate)
    #
    #     final_capture_rate = capture_rate * interference
    #     return final_capture_rate

    # @staticmethod # todo: dit capture rate noemen? en dit is hetzelfde voor alle patches en moet global
    # def functional_response_mussel(attack_rate, mussel_density, prey_weight, max_intake_rate):
    #     """
    #     Functional response as described in WEBTICS. They converted
    #     the intake of stillman to a capture rate.
    #
    #     :param attack_rate:
    #     :param max_intake_rate:
    #     :param prey_weight: average dry mass weight of mussels (no size classes) in g
    #     :return: capture rate in # prey/ s
    #     """
    #
    #     # calculate handling time and capture rate
    #     handling_time = (prey_weight * 1000) / max_intake_rate # convert prey to mg
    #     capture_rate = (attack_rate * mussel_density) / (1 + attack_rate * handling_time * mussel_density)
    #     return capture_rate

    # @staticmethod #todo call this maximal_mussel_intake
    # def maximal_intake_rate(prey_weight):
    #     """Calculate maximal intake rate as described in WEBTICS (page 62)
    #
    #     :prey_weight in g
    #     :return max intake rate in mg/s
    #     """
    #
    #     # parameters todo: in parameter file
    #     mussel_intake_rate_A = 0.092  # parameters for max intake rate (plateau)
    #     mussel_intake_rate_B = 0.506
    #
    #     # calculate plateau/max intake rate
    #     max_intake_rate = mussel_intake_rate_A * (prey_weight * 1000) ** mussel_intake_rate_B
    #     return max_intake_rate
    #
    # def consume_mudflats_diet(self):
    #     """ Method that lets agent forage on mudflat (currently only cockles taken into account).
    #
    #     In this method the depletion of prey on a patch is also implemented.
    #
    #     :return: The amount of wet weight foraged is returned (in g).
    #     """
    #
    #     # get the capture rate of all prey on mudflat (different cockle sizes)
    #     capture_rate_kok1, capture_rate_kok2, capture_rate_kokmj, capture_rate_mac \
    #         = self.combined_capture_rate_cockle_macoma()
    #
    #     # wet weight intake rate (g/s)
    #     patch_wet_intake = capture_rate_kok1 * self.model.cockle_wet_weight[self.pos][0] \
    #                          + capture_rate_kok2 * self.model.cockle_wet_weight[self.pos][1]\
    #                          + capture_rate_kokmj * self.model.cockle_wet_weight[self.pos][2]\
    #                          + capture_rate_mac * self.model.macoma_wet_weight[self.pos]
    #     patch_wet_intake *= (1 - self.model.LeftOverShellfish)
    #
    #     # convert to intake rate of one time step
    #     conversion_s_to_timestep = self.model.resolution_min * 60 # todo: dubbel
    #     total_patch_intake_wet_weight = patch_wet_intake * conversion_s_to_timestep
    #
    #     # calculate possible intake based on stomach left and digestive rate
    #     possible_wtw_intake = self.calculate_possible_intake()  # g / 10 minutes
    #
    #     # intake is minimum of possible intake and intake achievable on patch
    #     intake_wtw = min(total_patch_intake_wet_weight, possible_wtw_intake)   # WtW intake in g
    #
    #     # compare final intake to original patch intake todo: dit is nu wat moeilijker, mss met fracties?
    #     if total_patch_intake_wet_weight > 0:
    #         fraction_possible_final_intake = intake_wtw / total_patch_intake_wet_weight
    #     else:
    #         fraction_possible_final_intake = 0
    #
    #     # calculate final number of prey eaten
    #     final_captured_kok1 = capture_rate_kok1 * conversion_s_to_timestep * fraction_possible_final_intake
    #     final_captured_kok2 = capture_rate_kok2 * conversion_s_to_timestep * fraction_possible_final_intake
    #     final_captured_kokmj = capture_rate_kokmj * conversion_s_to_timestep * fraction_possible_final_intake
    #     final_captured_mac = capture_rate_mac * conversion_s_to_timestep * fraction_possible_final_intake
    #
    #     # calculate energy intake
    #     energy_intake = intake_wtw * self.model.FractionTakenUp * self.model.RatioAFDWtoWet \
    #                                 * self.model.AFDWenergyContent
    #
    #     # check if energy gain does not exceed goal, if so, adapt intake #todo: in functie?
    #     if self.energy_gain + energy_intake > self.energy_goal:
    #         # calculate surplus
    #         surplus = self.energy_gain + energy_intake - self.energy_goal
    #
    #         # fraction of this time step needed to accomplish goal
    #         fraction_needed = 1 - (surplus / energy_intake)
    #
    #         # multiply all intakes with fraction needed
    #         intake_wtw *= fraction_needed
    #         energy_intake *= fraction_needed
    #         final_captured_kok1 *= fraction_needed
    #         final_captured_kok2 *= fraction_needed
    #         final_captured_kokmj *= fraction_needed
    #         final_captured_mac *= fraction_needed
    #
    #         # update foraging time
    #         self.time_foraged += fraction_needed
    #     else:
    #         self.time_foraged += 1
    #
    #     # deplete prey (use actual area of patch)
    #     self.model.cockle_densities[self.pos][0] -= final_captured_kok1 / self.model.patch_areas[self.pos]
    #     self.model.cockle_densities[self.pos][1] -= final_captured_kok2 / self.model.patch_areas[self.pos]
    #     self.model.cockle_densities[self.pos][2] -= final_captured_kokmj / self.model.patch_areas[self.pos]
    #     self.model.macoma_density[self.pos] -= final_captured_mac / self.model.patch_areas[self.pos]
    #     return intake_wtw, energy_intake

    # def combined_capture_rate_cockle_macoma(self):
    #     """ Method that calculates the intake rate when agent forages on cockles. Three different size classes of
    #     cockles are taken into account (0-1, 2 and >2 years old)
    #
    #     The method looks at the density and weight of the cockles on the patch the agent is currently on and returns
    #     the capture rate of the different size class cockles in #/s.
    #     """
    #
    #     # get density and size of all cockle size classes on patch
    #     kok1_handling_time = self.model.handling_time_cockles[self.pos][0] # todo: one liner van maken
    #     kok2_handling_time = self.model.handling_time_cockles[self.pos][1]
    #     kokmj_handling_time = self.model.handling_time_cockles[self.pos][2]
    #     kok1_density, kok2_density, kokmj_density = self.model.cockle_densities[self.pos]
    #
    #     # macoma
    #     mac_density = self.model.macoma_density[self.pos]
    #     mac_handling_time = self.model.handling_time_macoma[self.pos]
    #
    #     # parameters
    #     leoA = 0.000860373  # Zwarts et al. (1996b), taken from WEBTICS
    #     leoB = 0.220524  # Zwarts et al.(1996b)
    #     hiddinkA = 0.000625 # Hiddink2003
    #     attack_rate = leoA * leoB
    #
    #     # calculate capture rate for every size class (number of cockles/s)
    #     capture_rate_kok1_num = attack_rate * kok1_density # numerator of eq 5.9 webtics
    #     capture_rate_kok1_den = attack_rate * kok1_handling_time * kok1_density # denominator without 1 +
    #     capture_rate_kok2_num = attack_rate * kok2_density
    #     capture_rate_kok2_den = attack_rate * kok2_handling_time * kok2_density
    #     capture_rate_kokmj_num = attack_rate * kokmj_density
    #     capture_rate_kokmj_den = attack_rate * kokmj_handling_time * kokmj_density
    #
    #     # capture rate macoma
    #     capture_rate_mac_num = hiddinkA * mac_density * self.model.proportion_macoma # only take available macoma into account
    #     capture_rate_mac_den = capture_rate_mac_num * mac_handling_time
    #
    #     # final denominator 5.9 webtics
    #     final_denominator = 1 + capture_rate_kok1_den + capture_rate_kok2_den + capture_rate_kokmj_den \
    #                         + capture_rate_mac_den
    #
    #     ##############
    #     # for cockles, calculate uptake reduction todo: dit moet binnen agent, rest van capture rate global maken
    #     bird_density = (self.model.num_agents_on_patches[self.pos] - 1) / self.model.available_areas[
    #                 self.pos]
    #
    #     # parameters
    #     attack_distance = 2.0  # webtics, stillman 2002
    #     alpha = 0.4  # fitted parameter by webtics
    #     relative_intake = self.calculate_cockle_uptake_reduction(bird_density, attack_distance, alpha)
    #
    #     # calculate number of captured prey for each size class todo: kan dit niet in betere vectorberekening?
    #     capture_rate_kok1 = (capture_rate_kok1_num / final_denominator) * relative_intake
    #     capture_rate_kok2 = (capture_rate_kok2_num / final_denominator) * relative_intake
    #     capture_rate_kokmj = (capture_rate_kokmj_num / final_denominator) * relative_intake
    #     capture_rate_mac = capture_rate_mac_num / final_denominator
    #     #################
    #     return capture_rate_kok1, capture_rate_kok2, capture_rate_kokmj, capture_rate_mac

    # def calculate_local_dominance(self, model):
    #     """
    #     Method that calculates local dominance (# of encounters won in %) for patch agent is currently on
    #
    #     Returns number of other agents on same patch and number of encounters won (L)
    #
    #     Higher dominance number means more dominance.
    #     """
    #
    #     # find dominance of all agents on same patch (excluding self)
    #     dominance_agents_same_patch = [agent.dominance for agent in model.agents_on_patches[self.pos]
    #                             if agent.unique_id != self.unique_id]
    #
    #     # calculate number of encounters won
    #     number_of_encounters = len(dominance_agents_same_patch) #todo: hier mss gewoon num_agents_patches van model pakken?
    #     if number_of_encounters == 0:
    #         L = 0
    #     else:
    #         agents_with_lower_dominance = [item for item in dominance_agents_same_patch if item < self.dominance] #todo: smaller then or equal?
    #         L = (len(agents_with_lower_dominance) / number_of_encounters) * 100
    #     return len(dominance_agents_same_patch), L

    # def consume_grassland_diet(self):
    #     """ Method that lets agent forage on grassland patch. Based on the energy goal and the stomach content
    #     the intake of an agent is evaluated.
    #
    #     The patch depletion is also implemented.
    #
    #     Returns the wet weight consumed (g).
    #     """
    #
    #     # parameters
    #     conversion_afdw_wtw = 0.17 # conversion from thesis Jeroen Onrust
    #
    #     # intake from Stillman (also used in webtics) % 
    #     afdw_intake_grassland = (0.53 * 60 * self.model.resolution_min) / 1000 # g / time step 10 mins, Stillman2000
    #
    #     # wtw intake % 
    #     wtw_intake = afdw_intake_grassland * 1 / conversion_afdw_wtw # g / time step
    #
    #     # calculate possible wtw intake based on stomach left and digestive rate
    #     possible_wtw_intake = self.calculate_possible_intake()  # g / time step
    #
    #     # intake is minimum of possible intake and intake achievable on patch
    #     final_intake_wtw = min(wtw_intake, possible_wtw_intake) * self.model.FractionTakenUp # WtW intake in g
    #
    #     # calculate energy intake, multiply with fraction of possible intake divided by max intake
    #     energy_intake = (afdw_intake_grassland * self.model.AFDWenergyContent) * final_intake_wtw / wtw_intake # kJ
    #
    #     # check if energy gain does not exceed goal, if so, adapt intake 
    #     if self.energy_gain + energy_intake > self.energy_goal:
    #         # calculate surplus
    #         surplus = self.energy_gain + energy_intake - self.energy_goal
    #
    #         # fraction of this time step needed to accomplish goal
    #         fraction_needed = 1 - (surplus / energy_intake)
    #
    #         # multiply all intakes with fraction needed
    #         final_intake_wtw *= fraction_needed
    #         energy_intake *= fraction_needed
    #
    #         # update foraging time
    #         self.time_foraged += fraction_needed
    #     else:
    #         self.time_foraged += 1
    #
    #     return final_intake_wtw, energy_intake








