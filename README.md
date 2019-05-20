This is the code for Q-value Path Decomposition for Deep Multiagent Reinforcement Learning (NeurIPS 2019).

Requirements:
1. Tensorflow;
2. Tensorboard_easy for logging;
3. SMAC SCII, follow the instructions by https://github.com/oxwhirl/smac.

After installing the environment, add the following two functions into the file starcraft2.py.

    def get_obs_agent_without_view_restriction(self, agent_id):
        """Returns observation for agent_id.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        unit = self.get_unit_by_id(agent_id)

        nf_al = 4 + self.unit_type_bits
        nf_en = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally
            nf_en += 1 + self.shield_bits_enemy

        if self.obs_last_action:
            nf_al += self.n_actions

        nf_own = self.unit_type_bits
        if self.obs_own_health:
            nf_own += 1 + self.shield_bits_ally

        move_feats_len = self.n_actions_move
        if self.obs_pathing_grid:
            move_feats_len += self.n_obs_pathing
        if self.obs_terrain_height:
            move_feats_len += self.n_obs_height

        move_feats = np.zeros(move_feats_len, dtype=np.float32)
        enemy_feats = np.zeros((self.n_enemies, nf_en), dtype=np.float32)
        ally_feats = np.zeros((self.n_agents - 1, nf_al), dtype=np.float32)
        own_feats = np.zeros(nf_own, dtype=np.float32)
        # own_id = np.zeros(self.n_agents)
        # own_id[agent_id] = 1

        if unit.health > 0:  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)

            # Movement features
            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]

            ind = self.n_actions_move

            if self.obs_pathing_grid:
                move_feats[
                    ind : ind + self.n_obs_pathing
                ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.obs_terrain_height:
                move_feats[ind:] = self.get_surrounding_height(unit)

            # Enemy features
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if (
                    # dist < sight_range and e_unit.health > 0
                    e_unit.health > 0
                ):  # visible and alive
                    # Sight range > shoot range
                    enemy_feats[e_id, 0] = avail_actions[
                        self.n_actions_no_attack + e_id
                    ]  # available
                    enemy_feats[e_id, 1] = dist / sight_range  # distance
                    enemy_feats[e_id, 2] = (
                        e_x - x
                    ) / sight_range  # relative X
                    enemy_feats[e_id, 3] = (
                        e_y - y
                    ) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        enemy_feats[e_id, ind] = (
                            e_unit.health / e_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            enemy_feats[e_id, ind] = (
                                e_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(e_unit, False)
                        enemy_feats[e_id, ind + type_id] = 1  # unit type

            # Ally features
            al_ids = [
                al_id for al_id in range(self.n_agents) if al_id != agent_id
            ]
            for i, al_id in enumerate(al_ids):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)

                if (
                    # dist < sight_range and al_unit.health > 0
                    al_unit.health > 0
                ):  # visible and alive
                    ally_feats[i, 0] = 1  # visible
                    ally_feats[i, 1] = dist / sight_range  # distance
                    ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                    ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        ally_feats[i, ind] = (
                            al_unit.health / al_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            ally_feats[i, ind] = (
                                al_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True)
                        ally_feats[i, ind + type_id] = 1
                        ind += self.unit_type_bits

                    if self.obs_last_action:
                        ally_feats[i, ind:] = self.last_action[al_id]

            # Own features
            ind = 0
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1

            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit, True)
                own_feats[ind + type_id] = 1

            # Agent id
            # own_id = np.zeros(self.n_agents)
            # own_id[agent_id] = 1

        agent_obs = np.concatenate(
            (
                move_feats.flatten(),
                enemy_feats.flatten(),
                ally_feats.flatten(),
                own_feats.flatten(),
                # own_id.flatten(),
            )
        )

        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug("Avail. actions {}".format(
                self.get_avail_agent_actions(agent_id)))
            logging.debug("Move feats {}".format(move_feats))
            logging.debug("Enemy feats {}".format(enemy_feats))
            logging.debug("Ally feats {}".format(ally_feats))
            logging.debug("Own feats {}".format(own_feats))

        return agent_obs

    def get_obs_without_view_restriction(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent_without_view_restriction(i) for i in range(self.n_agents)]
        return agents_obs

The two functions are used for obtaining agent observations without view restriction for the centralized critic.

For running in the Windows, use run.py.
For running in the Linux, use command line such as 'python main.py -s 0 -mn 3m'. (Follow the hyper-parameters as in the run.py)

For load/save models, see the options in file qpd_utils.py.
