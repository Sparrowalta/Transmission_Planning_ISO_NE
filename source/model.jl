function build_model(grid_data, wind_profile, pv_profile, load_profile, wind_node_count, pv_node_count, time_steps_per_scenario)
    buses = grid_data.buses
    lines = grid_data.lines
    generators = grid_data.generators
    num_buses = grid_data.num_buses
    num_lines = grid_data.num_lines
    num_generators = grid_data.num_generators
    slack_bus_idx = grid_data.slack_bus_index
    global MVA_BASE = 1 

    global time_step_resolution = 1 # hr 
    if enable_annual_resolution
        num_epochs = 19
        epoch_to_start_year = Dict(i => i for i in 1:19)
        epoch_to_end_year = Dict(i => i + 1 for i in 1:19)
    else
        num_epochs = 4 
        epoch_to_start_year = Dict(1 => 1, 2 => 6, 3 => 11, 4 => 16)
        epoch_to_end_year = Dict(1 => 5, 2 => 10, 3 => 15, 4 => 20)
    end

    global epoch_indices = collect(1:num_epochs)

    num_new_osw_lines = num_lines - num_existing_lines
    num_new_lines = enable_offshore_wind ? num_new_osw_lines : 0

    if !(num_new_osw_lines in [0, 17, 51]) 
        error("num_new_osw_lines must be one of [0, 17, 51] to be consistent with input files")
        stop()
    end

    if enable_offshore_wind & (num_new_osw_lines == 17)
        if enable_annual_resolution
            wind_online_year_map =  [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 4, 2, 1, 1]
        else
            wind_online_year_map = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        end
    elseif enable_offshore_wind & (num_new_osw_lines == 51)
        if enable_annual_resolution
            wind_online_year_map = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 4, 2, 1]
        else
            wind_online_year_map = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        end
    end

    _wind_power_at_node = zeros(num_buses, num_scenarios, time_steps_per_scenario)
    wind_power_at_node = zeros(num_planning_years, num_buses, num_scenarios, time_steps_per_scenario)
    @info "num_planning_years = $num_planning_years, num_buses = $num_buses, num_scenarios = $num_scenarios, time_steps_per_scenario = $time_steps_per_scenario"
    for y = 1:num_planning_years, n = 1:num_buses, e = 1:num_scenarios, t = 1:time_steps_per_scenario
            if enable_offshore_wind
                _wind_power_at_node[n, e, t] = (y >= wind_online_year_map[n]) ? (wind_profile[n, t+(e-1)*time_steps_per_scenario]) : 0
            else
                _wind_power_at_node[n, e, t] = wind_profile[n, t+(e-1)*time_steps_per_scenario]
            end
            wind_power_at_node[y, :, :, :] = _wind_power_at_node[:, :, :] + (y - 1) * 0.00 * _wind_power_at_node[:, :, :]
    end
    global final_wind_power = wind_power_at_node[[epoch_to_end_year[y] for y in epoch_indices], :, :, :]

    _pv_power_at_node = zeros(num_buses, num_scenarios, time_steps_per_scenario)
    pv_power_at_node = zeros(num_planning_years, num_buses, num_scenarios, time_steps_per_scenario)
    for y = 1:num_planning_years, n = 1:num_buses, e = 1:num_scenarios, t = 1:time_steps_per_scenario
            _pv_power_at_node[n, e, t] = pv_profile[n, t+(e-1)*time_steps_per_scenario]
            pv_power_at_node[y, :, :, :] = _pv_power_at_node[:, :, :] + (y - 1) * 0.00 * _pv_power_at_node[:, :, :]
    end
    global final_pv_power = pv_power_at_node[[epoch_to_end_year[y] for y in epoch_indices], :, :, :]

    wind_power_at_node = wind_power_at_node + pv_power_at_node

    _load_at_node = zeros(num_buses, num_scenarios, time_steps_per_scenario)
    load_at_node = zeros(num_planning_years, num_buses, num_scenarios, time_steps_per_scenario)
    for y = 1:num_planning_years, n = 1:num_buses, e = 1:num_scenarios, t = 1:time_steps_per_scenario
            _load_at_node[n, e, t] = load_profile[n, t+(e-1)*time_steps_per_scenario]
            load_at_node[y, :, :, :] = _load_at_node[:, :, :] .* ((1 + 0.023)^(y-1))
    end
    global final_load = load_at_node[[epoch_to_end_year[y] for y in epoch_indices], :, :, :]

    wind_profile_new = Matrix(CSV.read(joinpath(data_directory, "wind_normalized.csv"), DataFrame, header=false))'
    pv_profile_new = Matrix(CSV.read(joinpath(data_directory, "solar_normalized.csv"), DataFrame, header=false))'

    _wind_node_new_normalized = zeros(num_buses, num_scenarios, time_steps_per_scenario)
    wind_node_new_normalized = zeros(num_planning_years, num_buses, num_scenarios, time_steps_per_scenario)
    for y = 1:num_planning_years, n = 1:num_buses, e = 1:num_scenarios, t = 1:time_steps_per_scenario
            _wind_node_new_normalized[n, e, t] = wind_profile_new[n, t+(e-1)*time_steps_per_scenario]
            wind_node_new_normalized[y, :, :, :] = _wind_node_new_normalized[:, :, :] + (y - 1) * 0.00 * _wind_node_new_normalized[:, :, :]
    end
    global final_new_wind_profile = wind_node_new_normalized[[epoch_to_end_year[y] for y in epoch_indices], :, :, :]

    _pv_node_new_normalized = zeros(num_buses, num_scenarios, time_steps_per_scenario)
    pv_node_new_normalized = zeros(num_planning_years, num_buses, num_scenarios, time_steps_per_scenario)
    for y = 1:num_planning_years, n = 1:num_buses, e = 1:num_scenarios, t = 1:time_steps_per_scenario
            _pv_node_new_normalized[n, e, t] = pv_profile_new[n, t+(e-1)*time_steps_per_scenario]
            pv_node_new_normalized[y, :, :, :] = _pv_node_new_normalized[:, :, :] + (y - 1) * 0.00 * _pv_node_new_normalized[:, :, :]
    end
    global final_new_pv_profile = pv_node_new_normalized[[epoch_to_end_year[y] for y in epoch_indices], :, :, :]

    enable_ramping_constraints = true
    enable_reserve_constraints = true 

    days_in_year = 365
    _VoWS_per_bus = vcat(value_of_wind_spillage * ones(num_existing_buses), value_of_offshore_wind_spillage * ones(num_buses - num_existing_buses))
    time_step_indices = collect(1:time_steps_per_scenario)
    bus_indices = collect(1:num_buses)
    existing_bus_indices = bus_indices[1:num_existing_buses]
    new_bus_indices = bus_indices[num_existing_buses+1:end]
    line_indices = collect(1:num_lines)
    generator_indices = collect(1:num_generators)
    scenario_indices = collect(1:num_scenarios)
    candidate_line_indices = collect(1:num_existing_lines+num_new_lines)
    existing_line_indices = collect(1:num_existing_lines)
    upgradeable_line_indices = existing_line_indices
    new_line_indices = upgradeable_line_indices[end] .+ collect(1:num_new_lines)
    cable_type_indices = collect(1:3)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)
    set_optimizer_attribute(model, "Seed", 1234)
    set_optimizer_attribute(model, "Method", 1) 
    set_optimizer_attribute(model, "MIPGap", mip_optimality_gap)
    set_optimizer_attribute(model, "FeasibilityTol", 1e-9)
    set_optimizer_attribute(model, "OptimalityTol", 1e-9)

    # ============== #
    # Decision variables
    # ============== #
    @variable(model, line_build_start[line_indices, epoch_indices], Bin)
    @variable(model, line_in_service[line_indices, epoch_indices], Bin)
    
    @variable(model, thermal_generation[epoch_indices, generator_indices, scenario_indices, time_step_indices] >= 0)
    @variable(model, wind_generation[epoch_indices, bus_indices, scenario_indices, time_step_indices] >= 0)
    @variable(model, generator_reserve[epoch_indices, generator_indices, scenario_indices, time_step_indices] >= 0)
    @variable(model, battery_reserve[epoch_indices, bus_indices, scenario_indices, time_step_indices] >= 0)
    @variable(model, new_ng_cc_ccs_reserve[epoch_indices, bus_indices, scenario_indices, time_step_indices] >= 0)
    @variable(model, new_ng_ct_reserve[epoch_indices, bus_indices, scenario_indices, time_step_indices] >= 0)
    @variable(model, load_shedding[epoch_indices, bus_indices, scenario_indices, time_step_indices] >= 0)
    @variable(model, load_shedding_extreme[epoch_indices, bus_indices, scenario_indices, time_step_indices] >= 0)
    @variable(model, renewable_curtailment[epoch_indices, bus_indices, scenario_indices, time_step_indices] >= 0)
    @variable(model, voltage_angle[epoch_indices, bus_indices, scenario_indices, time_step_indices])
    @variable(model, bus_power_injection[epoch_indices, bus_indices, scenario_indices, time_step_indices])
    @variable(model, dc_power_flow[epoch_indices, line_indices, scenario_indices, time_step_indices])
    @variable(model, new_dc_power_flow[epoch_indices, line_indices, scenario_indices, time_step_indices])
    @variable(model, new_dc_power_flow_osw[epoch_indices, line_indices, scenario_indices, time_step_indices])
    @variable(model, new_wind_capacity[bus_indices, epoch_indices] >= 0)
    @variable(model, new_pv_capacity[bus_indices, epoch_indices] >= 0)
    @variable(model, new_ng_cc_ccs_capacity[bus_indices, epoch_indices] >= 0)
    @variable(model, new_ng_cc_ccs_generation[epoch_indices, bus_indices, scenario_indices, time_step_indices] >= 0)
    @variable(model, new_ng_ct_capacity[bus_indices, epoch_indices] >= 0)
    @variable(model, new_ng_ct_generation[epoch_indices, bus_indices, scenario_indices, time_step_indices] >= 0)
    @variable(model, battery_charge[epoch_indices, bus_indices, scenario_indices, time_step_indices] >= 0)
    @variable(model, battery_discharge[epoch_indices, bus_indices, scenario_indices, time_step_indices] >= 0)
    @variable(model, battery_state_of_charge[epoch_indices, bus_indices, scenario_indices, time_step_indices] >= 0)
    @variable(model, bess_energy_capacity[bus_indices, epoch_indices] >= 0)
    @variable(model, bess_power_capacity[bus_indices, epoch_indices] >= 0)
    @variable(model, available_power_capacity[bus_indices, epoch_indices] >= 0)
    @variable(model, available_energy_capacity[bus_indices, epoch_indices] >= 0)
    @variable(model, min_bess_energy_capacity[bus_indices, epoch_indices] >= 0)
    @variable(model, rps_noncompliance[epoch_indices, bus_indices] >= 0)
    @variable(model, osw_line_build_start[line_indices, cable_type_indices, epoch_indices], Bin)
    @variable(model, osw_line_in_service[line_indices, cable_type_indices, epoch_indices], Bin)

    if ~enable_offshore_battery
        @constraint(model, [s = new_bus_indices, y = epoch_indices], bess_power_capacity[s, y] == 0)
        @constraint(model, [s = new_bus_indices, y = epoch_indices], bess_energy_capacity[s, y] == 0)
    end
    @constraint(model, [s = new_bus_indices, y = epoch_indices], new_wind_capacity[s, y] == 0)
    @constraint(model, [s = new_bus_indices, y = epoch_indices], new_pv_capacity[s, y] == 0)
    @constraint(model, [s = new_bus_indices, y = epoch_indices], new_ng_cc_ccs_capacity[s, y] == 0)
    @constraint(model, [s = new_bus_indices, y = epoch_indices], new_ng_ct_capacity[s, y] == 0)

    # ============== #
    # Objective function
    # ============== #
    cost_data_atb_2022 = CSV.read(joinpath(data_directory, "costs_atb_2022.csv"), DataFrame)
    vom_ng_cc_ccs = cost_data_atb_2022[!, :vom_ng_cc_ccs]
    fom_ng_cc_ccs = cost_data_atb_2022[!, :fom_ng_cc_ccs] * 1000
    vom_ng_ct = cost_data_atb_2022[!, :vom_ng_ct]
    fom_ng_ct = cost_data_atb_2022[!, :fom_ng_ct] * 1000
    fom_wind = cost_data_atb_2022[!, :fom_wind] * 1000
    ptc_wind = cost_data_atb_2022[!, :ptc_wind]
    ptc_wind = enable_wind_ptc ? ptc_wind : ptc_wind * 0
    fom_pv = cost_data_atb_2022[!, :fom_pv] * 1000
    itc_pv = cost_data_atb_2022[!, :itc_pv]
    itc_pv = enable_pv_itc ? itc_pv : itc_pv * 0
    fom_battery = cost_data_atb_2022[!, :fom_battery] * 1000

    scenario_weights_raw = Matrix(CSV.read(joinpath(data_directory, "weight_of_scenerios.csv"),DataFrame, header=false))
    normal_scenario_weights = scenario_weights_raw[1:Int(length(scenario_weights_raw)/2)]
    extreme_scenario_weights = scenario_weights_raw[Int(length(scenario_weights_raw)/2)+1:end]
    
    scenario_weights = []
    if enable_extreme_scenarios
        if extreme_scenario_weight == -1.0
            scenario_weights = scenario_weights_raw
        else
            normal_scenario_weights = normal_scenario_weights .+ extreme_scenario_weights
            extreme_scenario_weights = extreme_scenario_weights ./= sum(extreme_scenario_weights)
            scenario_weights = vcat((1 - extreme_scenario_weight) .* normal_scenario_weights, extreme_scenario_weight .* extreme_scenario_weights)
        end
    else
        scenario_weights = normal_scenario_weights .+ extreme_scenario_weights
    end
    hours_per_scenario = days_in_year * scenario_weights

    if enable_demand_flexibility
        c1_flex = 400 
        c2_flex = 0.0035
        P_max_flex = 300
        @variable(model, demand_flex_amount[epoch_indices, bus_indices, scenario_indices, time_step_indices])
        @variable(model, demand_flex_block1[epoch_indices, bus_indices, scenario_indices, time_step_indices])
        @variable(model, demand_flex_block2[epoch_indices, bus_indices, scenario_indices, time_step_indices])
        @variable(model, demand_flex_block3[epoch_indices, bus_indices, scenario_indices, time_step_indices])
        @variable(model, demand_flex_block4[epoch_indices, bus_indices, scenario_indices, time_step_indices])
        @variable(model, abs_demand_flex_block1[epoch_indices, bus_indices, scenario_indices, time_step_indices] >= 0)
        @variable(model, abs_demand_flex_block2[epoch_indices, bus_indices, scenario_indices, time_step_indices] >= 0)
        @variable(model, abs_demand_flex_block3[epoch_indices, bus_indices, scenario_indices, time_step_indices] >= 0)
        @variable(model, abs_demand_flex_block4[epoch_indices, bus_indices, scenario_indices, time_step_indices] >= 0)
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], abs_demand_flex_block1[y, s, e, t] >= -demand_flex_block1[y, s, e, t])
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], abs_demand_flex_block1[y, s, e, t] >= demand_flex_block1[y, s, e, t])
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], abs_demand_flex_block2[y, s, e, t] >= -demand_flex_block2[y, s, e, t])
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], abs_demand_flex_block2[y, s, e, t] >= demand_flex_block2[y, s, e, t])
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], abs_demand_flex_block3[y, s, e, t] >= -demand_flex_block3[y, s, e, t])
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], abs_demand_flex_block3[y, s, e, t] >= demand_flex_block3[y, s, e, t])
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], abs_demand_flex_block4[y, s, e, t] >= -demand_flex_block4[y, s, e, t])
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], abs_demand_flex_block4[y, s, e, t] >= demand_flex_block4[y, s, e, t])
        
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], -demand_flexibility_factor * load_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE / 4 <= demand_flex_block1[y, s, e, t] <= demand_flexibility_factor * load_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE / 4)
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], -demand_flexibility_factor * load_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE / 4 <= demand_flex_block2[y, s, e, t] <= demand_flexibility_factor * load_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE / 4)
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], -demand_flexibility_factor * load_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE / 4 <= demand_flex_block3[y, s, e, t] <= demand_flexibility_factor * load_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE / 4)
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], -demand_flexibility_factor * load_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE / 4 <= demand_flex_block4[y, s, e, t] <= demand_flexibility_factor * load_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE / 4)
        
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], demand_flex_amount[y, s, e, t] == demand_flex_block1[y, s, e, t] + demand_flex_block2[y, s, e, t] + demand_flex_block3[y, s, e, t] + demand_flex_block4[y, s, e, t])
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], - demand_flexibility_factor * load_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE <= demand_flex_amount[y, s, e, t] <= demand_flexibility_factor * load_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE)
        
        @expression(model, annual_operating_cost[y = epoch_indices], (1 / ((1 + discount_rate)^(epoch_to_end_year[y] - 1))) * (sum(hours_per_scenario[e] *
            (sum(((generators[i].cost_coeff_linear * thermal_generation[y, i, e, t] * MVA_BASE)) for i in generator_indices,t in time_step_indices)
                + sum((_VoWS_per_bus[s] * renewable_curtailment[y, s, e, t] * MVA_BASE) for s in bus_indices,t in time_step_indices)
                + sum((value_of_lost_load * load_shedding[y, s, e, t] * MVA_BASE) for s in bus_indices,t in time_step_indices)
                + (enable_extreme_scenarios ? sum((value_of_lost_load_extreme * load_shedding_extreme[y, s, e, t] * MVA_BASE) for s in bus_indices,t in time_step_indices; init = 0) : 0)
                + sum(vom_ng_cc_ccs[epoch_to_end_year[r]] * new_ng_cc_ccs_generation[r, s, e, t] * MVA_BASE for r in 1:y, s in bus_indices, t in time_step_indices)
                + sum(vom_ng_ct[epoch_to_end_year[r]] * new_ng_ct_generation[r, s, e, t] * MVA_BASE for r in 1:y, s in bus_indices, t in time_step_indices)
                + sum(new_wind_capacity[s, y] * MVA_BASE * wind_node_new_normalized[epoch_to_end_year[y], s, e, t] * 0 for s in bus_indices, t in time_step_indices)
                - (enable_wind_ptc ? (sum(new_wind_capacity[s, y] * wind_node_new_normalized[epoch_to_end_year[y], s, e, t] * ptc_wind[y] for s in bus_indices, t in time_step_indices)) : 0)
                + sum(new_pv_capacity[s, y] * MVA_BASE * pv_node_new_normalized[epoch_to_end_year[y], s, e, t] * 0 for s in bus_indices, t in time_step_indices)
                + sum(value_of_lost_load * abs_demand_flex_block4[y, s, e, t] * MVA_BASE + (value_of_lost_load - (c1_flex + 2 * c2_flex * P_max_flex))/4 * abs_demand_flex_block3[y, s, e, t] * MVA_BASE + (value_of_lost_load - (c1_flex + 2 * c2_flex * P_max_flex))/8 * abs_demand_flex_block2[y, s, e, t] * MVA_BASE + (value_of_lost_load - (c1_flex + 2 * c2_flex * P_max_flex))/12 * abs_demand_flex_block1[y, s, e, t] * MVA_BASE for s in bus_indices,t in time_step_indices)) for e in scenario_indices)
            + sum(fom_ng_cc_ccs[epoch_to_end_year[y]] * new_ng_cc_ccs_capacity[s, y] * MVA_BASE for s in bus_indices)
            + sum(fom_ng_ct[epoch_to_end_year[y]] * new_ng_ct_capacity[s, y] * MVA_BASE for s in bus_indices)
            + sum(fom_wind[epoch_to_end_year[y]] * new_wind_capacity[s, y] * MVA_BASE for s in bus_indices)
            + sum(fom_pv[epoch_to_end_year[y]] * new_pv_capacity[s, y] * MVA_BASE for s in bus_indices)
            + (enable_battery_storage ? sum(fom_battery[epoch_to_end_year[y]] * bess_power_capacity[s, y] * MVA_BASE for s in bus_indices; init = 0) : 0)
            + sum((rps_noncompliance_penalty_cost * rps_noncompliance[y, s] * MVA_BASE) for s in bus_indices))
        )
        @expression(model, over_generation_penalty[y = epoch_indices], (1 / ((1 + discount_rate)^(epoch_to_end_year[y] - 1))) * sum(hours_per_scenario[e] * (sum((_VoWS_per_bus[s] * renewable_curtailment[y, s, e, t] * MVA_BASE) for s in bus_indices,t in time_step_indices)) for e in scenario_indices))
        @expression(model, under_generation_penalty[y = epoch_indices], (1 / ((1 + discount_rate)^(epoch_to_end_year[y] - 1))) * sum(hours_per_scenario[e] * (sum((value_of_lost_load * load_shedding[y, s, e, t] * MVA_BASE + value_of_lost_load_extreme * load_shedding_extreme[y, s, e, t] * MVA_BASE) for s in bus_indices,t in time_step_indices)) for e in scenario_indices))
        @expression(model, demand_flexibility_cost[y = epoch_indices], (1 / ((1 + discount_rate)^(epoch_to_end_year[y] - 1))) * sum(hours_per_scenario[e] * (sum(value_of_lost_load * abs_demand_flex_block4[y, s, e, t] * MVA_BASE + (value_of_lost_load - (c1_flex + 2 * c2_flex * P_max_flex))/4 * abs_demand_flex_block3[y, s, e, t] * MVA_BASE + (value_of_lost_load - (c1_flex + 2 * c2_flex * P_max_flex))/8 * abs_demand_flex_block2[y, s, e, t] * MVA_BASE + (value_of_lost_load - (c1_flex + 2 * c2_flex * P_max_flex))/12 * abs_demand_flex_block1[y, s, e, t] * MVA_BASE for s in bus_indices,t in time_step_indices)) for e in scenario_indices))
        @expression(model, rps_noncompliance_penalty[y = epoch_indices], (1 / ((1 + discount_rate)^(epoch_to_end_year[y] - 1))) * sum((rps_noncompliance_penalty_cost * rps_noncompliance[y, s] * MVA_BASE) for s in bus_indices))
    else
        @expression(model, annual_operating_cost[y = epoch_indices], (1 / ((1 + discount_rate)^(epoch_to_end_year[y] - 1))) * (sum(hours_per_scenario[e] *
            (sum(((generators[i].cost_coeff_linear * thermal_generation[y, i, e, t] * MVA_BASE)) for i in generator_indices,t in time_step_indices)
                + sum((_VoWS_per_bus[s] * renewable_curtailment[y, s, e, t] * MVA_BASE) for s in bus_indices,t in time_step_indices)
                + sum((value_of_lost_load * load_shedding[y, s, e, t] * MVA_BASE) for s in bus_indices,t in time_step_indices)
                + (enable_extreme_scenarios ? sum((value_of_lost_load_extreme * load_shedding_extreme[y, s, e, t] * MVA_BASE) for s in bus_indices,t in time_step_indices; init = 0) : 0)
                + sum(vom_ng_cc_ccs[epoch_to_end_year[r]] * new_ng_cc_ccs_generation[r, s, e, t] * MVA_BASE for r in 1:y, s in bus_indices, t in time_step_indices)
                + sum(vom_ng_ct[epoch_to_end_year[r]] * new_ng_ct_generation[r, s, e, t] * MVA_BASE for r in 1:y, s in bus_indices, t in time_step_indices)
                + sum(new_wind_capacity[s, y] * MVA_BASE * wind_node_new_normalized[epoch_to_end_year[y], s, e, t] * 0 for s in bus_indices, t in time_step_indices)
                - (enable_wind_ptc ? (sum(new_wind_capacity[s, y] * MVA_BASE * wind_node_new_normalized[epoch_to_end_year[y], s, e, t] * ptc_wind[y] for s in bus_indices, t in time_step_indices)) : 0)
                + sum(new_pv_capacity[s, y] * MVA_BASE * pv_node_new_normalized[epoch_to_end_year[y], s, e, t] * 0 for s in bus_indices, t in time_step_indices)) for e in scenario_indices)
            + sum(fom_ng_cc_ccs[epoch_to_end_year[y]] * new_ng_cc_ccs_capacity[s, y] * MVA_BASE for s in bus_indices)
            + sum(fom_ng_ct[epoch_to_end_year[y]] * new_ng_ct_capacity[s, y] * MVA_BASE for s in bus_indices)
            + sum(fom_wind[epoch_to_end_year[y]] * new_wind_capacity[s, y] * MVA_BASE for s in bus_indices)
            + sum(fom_pv[epoch_to_end_year[y]] * new_pv_capacity[s, y] * MVA_BASE for s in bus_indices)
            + (enable_battery_storage ? sum(fom_battery[epoch_to_end_year[y]] * bess_power_capacity[s, y] * MVA_BASE for s in bus_indices; init = 0) : 0) 
            + sum((rps_noncompliance_penalty_cost * rps_noncompliance[y, s] * MVA_BASE) for s in bus_indices))
        )
        @expression(model, over_generation_penalty[y = epoch_indices], (1 / ((1 + discount_rate)^(epoch_to_end_year[y] - 1))) * sum(hours_per_scenario[e] * (sum((_VoWS_per_bus[s] * renewable_curtailment[y, s, e, t] * MVA_BASE) for s in bus_indices,t in time_step_indices)) for e in scenario_indices))
        @expression(model, under_generation_penalty[y = epoch_indices], (1 / ((1 + discount_rate)^(epoch_to_end_year[y] - 1))) * sum(hours_per_scenario[e] * (sum((value_of_lost_load * load_shedding[y, s, e, t] * MVA_BASE) for s in bus_indices,t in time_step_indices)) for e in scenario_indices))
        @expression(model, rps_noncompliance_penalty[y = epoch_indices], (1 / ((1 + discount_rate)^(epoch_to_end_year[y] - 1))) * sum((rps_noncompliance_penalty_cost * rps_noncompliance[y, s] * MVA_BASE) for s in bus_indices))
    end

    bess_energy_capex = cost_data_atb_2022[!, :CE] * 1e3
    bess_power_capex = cost_data_atb_2022[!, :CP] * 1e3
    annualized_energy_cost = bess_energy_capex * discount_rate * (1 + discount_rate)^storage_lifetime / ((1 + discount_rate)^storage_lifetime - 1)
    annualized_power_cost = bess_power_capex * discount_rate * (1 + discount_rate)^storage_lifetime / ((1 + discount_rate)^storage_lifetime - 1)
    storage_discount_factor(n) = (n <= storage_lifetime) ? 1 / (1 + discount_rate)^(n - 1) : 0
    @expression(model, bess_investment_cost[y = epoch_indices], sum(storage_discount_factor(epoch_to_start_year[y-n+1]) * (sum(annualized_energy_cost[epoch_to_end_year[n]] * bess_energy_capacity[s,n] * MVA_BASE + annualized_power_cost[epoch_to_end_year[n]] * bess_power_capacity[s,n] * MVA_BASE for s in bus_indices)) for n in 1:y))

    line_lengths = CSV.read(joinpath(data_directory, "lines.csv"), DataFrame)[:, "Distance (miles)"]
    line_capacities = CSV.read(joinpath(data_directory, "lines.csv"), DataFrame)[:, "s_max"]
    if enable_offshore_wind
        offshore_cost_scaler =  0.7340821889389222
        cost_line_400 = offshore_cost_scaler * (0.007225 .* (line_lengths .* 1.609).^2 + 0.767 .* (line_lengths .* 1.609) .+ 32.8125) .* 1e6 .* (1.5285 * 1.09)
        cost_line_1400 = offshore_cost_scaler * (1.36 .* (line_lengths .* 1.609) .+ 366.78) .* 1e6 .* (1.5285 * 1.09)
        cost_line_2200 = offshore_cost_scaler * (1.8 .* (line_lengths .* 1.609) .+ 562.08) .* 1e6 .* (1.5285 * 1.09)

        annualized_cost_line_400 = cost_line_400 * discount_rate * (1 + discount_rate)^transmission_lifetime / ((1 + discount_rate)^transmission_lifetime - 1)
        annualized_cost_line_1400 = cost_line_1400 * discount_rate * (1 + discount_rate)^transmission_lifetime / ((1 + discount_rate)^transmission_lifetime - 1)
        annualized_cost_line_2200 = cost_line_2200 * discount_rate * (1 + discount_rate)^transmission_lifetime / ((1 + discount_rate)^transmission_lifetime - 1)
    else
        onshore_line_cost = onshore_line_cost_scaler * 3500 * 1.111 * line_capacities[1:num_existing_lines] .* line_lengths[1:num_existing_lines]
        line_lengths = line_lengths[1:num_existing_lines]
        annualized_onshore_line_cost = onshore_line_cost * discount_rate * (1 + discount_rate)^transmission_lifetime / ((1 + discount_rate)^transmission_lifetime - 1)
    end

    onshore_line_cost_override = onshore_line_cost_scaler * 3500 * 1.111 * line_capacities[1:num_existing_lines] .* line_lengths[1:num_existing_lines]
    annualized_onshore_line_cost = onshore_line_cost_override * discount_rate * (1 + discount_rate)^transmission_lifetime / ((1 + discount_rate)^transmission_lifetime - 1)
    transmission_discount_factor(n) = (n <= transmission_lifetime) ? 1 / (1 + discount_rate)^(n - 1) : 0
    
    @expression(model, line_investment_cost[y = epoch_indices], sum(transmission_discount_factor(epoch_to_start_year[y-n+1]) * sum(annualized_onshore_line_cost[l] * line_build_start[l, n] for l in upgradeable_line_indices) for n in 1:y)
                                        +
                                         (enable_offshore_wind ? (sum(transmission_discount_factor(epoch_to_start_year[y-n+1]) * sum(annualized_cost_line_400[l] * osw_line_build_start[l, 1, n] for l in new_line_indices) for n in 1:y) + sum(transmission_discount_factor(epoch_to_start_year[y-n+1]) * sum(annualized_cost_line_1400[l] * osw_line_build_start[l,2,n] for l in new_line_indices) for n in 1:y) + sum(transmission_discount_factor(epoch_to_start_year[y-n+1]) * sum(annualized_cost_line_2200[l] * osw_line_build_start[l,3,n] for l in new_line_indices) for n in 1:y)) : 0))

    capex_ng_cc_ccs = cost_data_atb_2022[!, :Cp_ng_cc_ccs] * 1000
    capex_ng_ct = cost_data_atb_2022[!, :Cp_ng_ct] * 1000
    capex_wind = cost_data_atb_2022[!, :Cp_wind] * 1000
    capex_pv = cost_data_atb_2022[!, :Cp_pv] * 1000
    capex_pv = -capex_pv .* (itc_pv .- 1)
    annualized_capex_ng_cc_ccs = capex_ng_cc_ccs * discount_rate * (1 + discount_rate)^generator_lifetime / ((1 + discount_rate)^generator_lifetime - 1)
    annualized_capex_ng_ct = capex_ng_ct * discount_rate * (1 + discount_rate)^generator_lifetime / ((1 + discount_rate)^generator_lifetime - 1)
    annualized_capex_wind = capex_wind * discount_rate * (1 + discount_rate)^generator_lifetime / ((1 + discount_rate)^generator_lifetime - 1)
    annualized_capex_pv = capex_pv * discount_rate * (1 + discount_rate)^generator_lifetime / ((1 + discount_rate)^generator_lifetime - 1)
    generator_discount_factor(n) = (n <= generator_lifetime) ? 1 / (1 + discount_rate)^(n - 1) : 0
    @expression(model, generator_investment_cost[y = epoch_indices], sum(generator_discount_factor(epoch_to_start_year[y-n+1]) * sum(annualized_capex_ng_cc_ccs[epoch_to_start_year[n]] * new_ng_cc_ccs_capacity[s, n] * MVA_BASE for s in bus_indices) for n in 1:y)
                                        + sum(generator_discount_factor(epoch_to_start_year[y-n+1]) * sum(annualized_capex_ng_ct[epoch_to_start_year[n]] * new_ng_ct_capacity[s, n] * MVA_BASE for s in bus_indices) for n in 1:y)
                                        + sum(generator_discount_factor(epoch_to_start_year[y-n+1]) * sum(annualized_capex_wind[epoch_to_start_year[n]] * new_wind_capacity[s, n] * MVA_BASE for s in bus_indices) for n in 1:y)
                                        + sum(generator_discount_factor(epoch_to_start_year[y-n+1]) * sum(annualized_capex_pv[epoch_to_start_year[n]] * new_pv_capacity[s, n] * MVA_BASE for s in bus_indices) for n in 1:y))

    emission_factor_co2_ng = 0.725072727
    emission_factor_so2_ng = 0.001663782
    emission_factor_nox_ng = 0.019212469
    emission_factor_pm25_ng = 0.0000251682
    
    @expression(model, co2_emission_cost[y = epoch_indices], (1 / ((1 + discount_rate)^(epoch_to_end_year[y] - 1))) * sum(hours_per_scenario[e] * (sum(co2_price * generators[i].co2_tons_per_mwh * thermal_generation[y, i, e, t] * MVA_BASE for i in generator_indices,t in time_step_indices)
        + sum(co2_price * emission_factor_co2_ng * new_ng_cc_ccs_generation[y, s, e, t] * MVA_BASE for s in bus_indices,t in time_step_indices)
        + sum(co2_price * emission_factor_co2_ng * new_ng_ct_generation[y, s, e, t] * MVA_BASE for s in bus_indices,t in time_step_indices)) for e in scenario_indices))
    
    @expression(model, air_quality_damage_cost[y = epoch_indices], (1 / ((1 + discount_rate)^(epoch_to_end_year[y] - 1))) * sum(hours_per_scenario[e] * (sum(generators[i].avg_marginal_damage * 1.08 * air_quality_cost_scaler * thermal_generation[y, i, e, t] * MVA_BASE for i in generator_indices,t in time_step_indices)
        + sum(22.415 * 1.08 * air_quality_cost_scaler * new_ng_cc_ccs_generation[y, s, e, t] * MVA_BASE for s in bus_indices,t in time_step_indices)
        + sum(26.22 * 1.08 * air_quality_cost_scaler * new_ng_ct_generation[y, s, e, t] * MVA_BASE for s in bus_indices,t in time_step_indices)) for e in scenario_indices))

    @expression(model, total_co2_emissions[y = epoch_indices], sum(hours_per_scenario[e] * (sum((generators[i].co2_tons_per_mwh * thermal_generation[y, i, e, t] * MVA_BASE) for i in generator_indices,t in time_step_indices) + emission_factor_co2_ng * sum(new_ng_cc_ccs_generation[r, s, e, t] * MVA_BASE for r in 1:y, s in bus_indices, t in time_step_indices) + emission_factor_co2_ng * sum(new_ng_ct_generation[r, s, e, t] * MVA_BASE for r in 1:y, s in bus_indices, t in time_step_indices)) for e in scenario_indices))
    @expression(model, total_so2_emissions[y = epoch_indices], sum(hours_per_scenario[e] * (sum((generators[i].so2_tons_per_mwh * thermal_generation[y, i, e, t] * MVA_BASE) for i in generator_indices,t in time_step_indices) + emission_factor_so2_ng * sum(new_ng_cc_ccs_generation[r, s, e, t] * MVA_BASE for r in 1:y, s in bus_indices, t in time_step_indices) + emission_factor_so2_ng * sum(new_ng_ct_generation[r, s, e, t] * MVA_BASE for r in 1:y, s in bus_indices, t in time_step_indices)) for e in scenario_indices))
    @expression(model, total_nox_emissions[y = epoch_indices], sum(hours_per_scenario[e] * (sum((generators[i].nox_tons_per_mwh * thermal_generation[y, i, e, t] * MVA_BASE) for i in generator_indices,t in time_step_indices) + emission_factor_nox_ng * sum(new_ng_cc_ccs_generation[r, s, e, t] * MVA_BASE for r in 1:y, s in bus_indices, t in time_step_indices) + emission_factor_nox_ng * sum(new_ng_ct_generation[r, s, e, t] * MVA_BASE for r in 1:y, s in bus_indices, t in time_step_indices)) for e in scenario_indices))
    @expression(model, total_pm25_emissions[y = epoch_indices], sum(hours_per_scenario[e] * (sum((generators[i].pm25_tons_per_mwh * thermal_generation[y, i, e, t] * MVA_BASE) for i in generator_indices,t in time_step_indices) + emission_factor_pm25_ng * sum(new_ng_cc_ccs_generation[r, s, e, t] * MVA_BASE for r in 1:y, s in bus_indices, t in time_step_indices) + emission_factor_pm25_ng * sum(new_ng_ct_generation[r, s, e, t] * MVA_BASE for r in 1:y, s in bus_indices, t in time_step_indices)) for e in scenario_indices))

    if enable_multi_objective
        @objective(model, Min, sum(annual_operating_cost[y] * objective_scaling_factor + line_investment_cost[y] * objective_scaling_factor + generator_investment_cost[y] * objective_scaling_factor + (enable_battery_storage ? bess_investment_cost[y] * objective_scaling_factor : 0) + (enable_emission_costs ? co2_emission_cost[y] * objective_scaling_factor : 0) + (enable_air_quality_costs ? air_quality_damage_cost[y] * objective_scaling_factor : 0) for y in epoch_indices))
    else
        @objective(model, Min, sum(annual_operating_cost[y] * objective_scaling_factor + line_investment_cost[y] * objective_scaling_factor + generator_investment_cost[y] * objective_scaling_factor + (enable_battery_storage ? bess_investment_cost[y] * objective_scaling_factor : 0) for y in epoch_indices))
    end

    # ======== Constraints ======== #
    @variable(model, is_generator_on[y in epoch_indices, i in generator_indices], Bin)
    new_gen_ramp_rate = 400
    for y in epoch_indices
        for e in scenario_indices, t in time_step_indices, i in generator_indices
            fix(is_generator_on[y, i], 1.0)
            @constraint(model, thermal_generation[y, i, e, t] >= 0 / MVA_BASE * is_generator_on[y, i])
            @constraint(model, thermal_generation[y, i, e, t] <= generators[i].max_power_output / MVA_BASE * is_generator_on[y, i])
        end
        for e in scenario_indices, t in time_step_indices, s in bus_indices
            @constraint(model, new_ng_cc_ccs_generation[y, s, e, t] >= 0)
            @constraint(model, new_ng_cc_ccs_generation[y, s, e, t] <= sum(new_ng_cc_ccs_capacity[s, r] / MVA_BASE for r ∈ 1:y))
            @constraint(model, new_ng_ct_generation[y, s, e, t] >= 0)
            @constraint(model, new_ng_ct_generation[y, s, e, t] <= sum(new_ng_ct_capacity[s, r] / MVA_BASE for r ∈ 1:y))
        end
        if enable_ramping_constraints
            for e in scenario_indices, t in time_step_indices[2:end], i in generator_indices
                @constraint(model, (thermal_generation[y, i, e, t] + generator_reserve[y, i, e, t] - thermal_generation[y, i, e, t-1]) <= generators[i].ramp_up_rate / MVA_BASE * is_generator_on[y, i])
                @constraint(model, -(thermal_generation[y, i, e, t] - generator_reserve[y, i, e, t-1] - thermal_generation[y, i, e, t-1]) <= generators[i].ramp_down_rate / MVA_BASE * is_generator_on[y, i])
            end
            for e in scenario_indices, t in time_step_indices[2:end], s in bus_indices
                @constraint(model, (new_ng_cc_ccs_generation[y, s, e, t] + new_ng_cc_ccs_reserve[y, s, e, t] - new_ng_cc_ccs_generation[y, s, e, t-1]) <= new_gen_ramp_rate / MVA_BASE)
                @constraint(model, -(new_ng_cc_ccs_generation[y, s, e, t] - new_ng_cc_ccs_reserve[y, s, e, t-1] - new_ng_cc_ccs_generation[y, s, e, t-1]) <= new_gen_ramp_rate / MVA_BASE)
                @constraint(model, (new_ng_ct_generation[y, s, e, t] + new_ng_ct_reserve[y, s, e, t] - new_ng_ct_generation[y, s, e, t-1]) <= new_gen_ramp_rate / MVA_BASE)
                @constraint(model, -(new_ng_ct_generation[y, s, e, t] - new_ng_ct_reserve[y, s, e, t-1] - new_ng_ct_generation[y, s, e, t-1]) <= new_gen_ramp_rate / MVA_BASE)
            end
        end
    end

    if enable_exogenous_retirements
        for i in generator_indices, y in epoch_indices, e in scenario_indices, t in time_step_indices
            if (generators[i].planned_retirement_year <= planning_start_year)
                fix(is_generator_on[y, i], 0.0; force=true)
            end
        end
        for i in generator_indices, y in epoch_indices, e in scenario_indices, t in time_step_indices
            if (generators[i].planned_retirement_year > planning_start_year) & (generators[i].planned_retirement_year <= planning_start_year + num_planning_years)
                if y < ((generators[i].planned_retirement_year - (planning_start_year-1)) ÷ (num_epochs+1)  + 1)
                    # pass
                else
                    fix(is_generator_on[y, i], 0.0; force=true)
                end
            end
        end
    end

    for y in epoch_indices
        for e in scenario_indices, t in time_step_indices, i in generator_indices
            @constraint(model, thermal_generation[y, i, e, t] + generator_reserve[y, i, e, t] <= generators[i].max_power_output / MVA_BASE * is_generator_on[y, i])
            if enable_ramping_constraints
                @constraint(model, generator_reserve[y, i, e, t] <= generators[i].ramp_up_rate / MVA_BASE * time_step_duration * is_generator_on[y, i])
            end
        end
        for e in scenario_indices, t in time_step_indices, s in bus_indices
            @constraint(model, new_ng_cc_ccs_generation[y, s, e, t] + new_ng_cc_ccs_reserve[y, s, e, t] <= sum(new_ng_cc_ccs_capacity[s, r] / MVA_BASE for r ∈ 1:y))
            if enable_ramping_constraints
                @constraint(model, new_ng_cc_ccs_reserve[y, s, e, t] <= new_gen_ramp_rate / MVA_BASE * time_step_duration)
            end
            @constraint(model, new_ng_ct_generation[y, s, e, t] + new_ng_ct_reserve[y, s, e, t] <= sum(new_ng_ct_capacity[s, r] / MVA_BASE for r ∈ 1:y))
            if enable_ramping_constraints
                @constraint(model, new_ng_ct_reserve[y, s, e, t] <= new_gen_ramp_rate / MVA_BASE * time_step_duration)
            end
        end
    end

    if enable_battery_storage
        @constraint(model, [s = bus_indices, y = epoch_indices], min_bess_energy_capacity[s, y] == 0.2 * bess_energy_capacity[s, y])
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices[2:end]], battery_state_of_charge[y, s, e, t] == battery_state_of_charge[y, s, e, t-1] +  battery_charging_efficiency * battery_charge[y, s, e, t] * time_step_resolution - battery_discharge[y, s, e, t] * time_step_resolution / battery_discharging_efficiency)
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices[1]], battery_state_of_charge[y, s, e, t] == 0.80 * available_energy_capacity[s, y])
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices[end]], battery_state_of_charge[y, s, e, t] == 0.80 * available_energy_capacity[s, y])
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], min_bess_energy_capacity[s, y] <= battery_state_of_charge[y, s, e, t])
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], battery_state_of_charge[y, s, e, t] <= available_energy_capacity[s, y])
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], battery_charge[y, s, e, t] <= available_power_capacity[s, y])
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], battery_discharge[y, s, e, t] <= available_power_capacity[s, y])
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], battery_reserve[y, s, e, t] + battery_discharge[y, s, e, t] - battery_charge[y, s, e, t] <= available_power_capacity[s, y])
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], (battery_reserve[y, s, e, t] - battery_charge[y, s, e, t]) * time_step_duration <= battery_discharging_efficiency * (battery_state_of_charge[y, s, e, t] - min_bess_energy_capacity[s,y]))

        degradation_factor = 1 - calendar_degradation_rate - cycle_degradation_rate
        energy_degradation(n) = (n <= storage_lifetime) ? 1 * (degradation_factor)^(n - 1) : 0
        @constraint(model, [s = bus_indices, y = epoch_indices], available_energy_capacity[s, y] == sum(energy_degradation(epoch_to_start_year[y-t₀+1]) * bess_energy_capacity[s, t₀] for t₀ in 1:y))
        power_degradation(n) = (n <= storage_lifetime) ? 1 : 0
        @constraint(model, [s = bus_indices, y = epoch_indices], available_power_capacity[s, y] == sum(power_degradation(epoch_to_start_year[y-t₀+1]) * bess_power_capacity[s, t₀] for t₀ in 1:y))
        @constraint(model, [s = bus_indices, y = epoch_indices], battery_duration_hours * bess_power_capacity[s, y] == bess_energy_capacity[s, y])
    end

    if ~enable_extreme_scenarios
        @info "Power balance when num_extreme_scenarios == 0: normal scenario_indices = $scenario_indices"
        @constraint(model, power_balance[y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], new_ng_cc_ccs_generation[y, s, e, t] + new_ng_ct_generation[y, s, e, t] + sum(thermal_generation[y, g, e, t] for g in buses[s].generators_at_bus) + (wind_power_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE - renewable_curtailment[y, s, e, t]) + sum(new_wind_capacity[s, r] * wind_node_new_normalized[epoch_to_end_year[y], s, e, t] / MVA_BASE + new_pv_capacity[s, r] * pv_node_new_normalized[epoch_to_end_year[y], s, e, t] / MVA_BASE for r in 1:y) == load_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE + (enable_demand_flexibility ? demand_flex_amount[y, s, e, t] : 0) - load_shedding[y, s, e, t] + bus_power_injection[y, s, e, t] + (enable_battery_storage ? (battery_charge[y, s, e, t] - battery_discharge[y, s, e, t]) : 0))
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices],  load_shedding[y, s, e, t] <= load_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE + (enable_demand_flexibility ? demand_flex_amount[y, s, e, t] : 0))
    else
        @info "Power balance num_extreme_scenarios == $num_extreme_scenarios: normal scenario_indices = $(scenario_indices[1:num_scenarios-num_extreme_scenarios])"
        @info "Power balance num_extreme_scenarios == $num_extreme_scenarios: extreme scenario_indices = $(scenario_indices[num_scenarios-num_extreme_scenarios+1:end])"
        @constraint(model, power_balance_normal[y = epoch_indices, s = bus_indices, e = scenario_indices[1:num_scenarios-num_extreme_scenarios], t = time_step_indices], new_ng_cc_ccs_generation[y, s, e, t] + new_ng_ct_generation[y, s, e, t] + sum(thermal_generation[y, g, e, t] for g in buses[s].generators_at_bus) + (wind_power_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE - renewable_curtailment[y, s, e, t]) + sum(new_wind_capacity[s, r] * wind_node_new_normalized[epoch_to_end_year[y], s, e, t] / MVA_BASE + new_pv_capacity[s, r] * pv_node_new_normalized[epoch_to_end_year[y], s, e, t] / MVA_BASE for r in 1:y) == load_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE + (enable_demand_flexibility ? demand_flex_amount[y, s, e, t] : 0) - load_shedding[y, s, e, t] + bus_power_injection[y, s, e, t] + (enable_battery_storage ? (battery_charge[y, s, e, t] - battery_discharge[y, s, e, t]) : 0))
        @constraint(model, power_balance_extreme[y = epoch_indices, s = bus_indices, e = scenario_indices[num_scenarios-num_extreme_scenarios+1:end], t = time_step_indices], new_ng_cc_ccs_generation[y, s, e, t] + new_ng_ct_generation[y, s, e, t] + sum(thermal_generation[y, g, e, t] for g in buses[s].generators_at_bus) + (wind_power_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE - renewable_curtailment[y, s, e, t]) + sum(new_wind_capacity[s, r] * wind_node_new_normalized[epoch_to_end_year[y], s, e, t] / MVA_BASE + new_pv_capacity[s, r] * pv_node_new_normalized[epoch_to_end_year[y], s, e, t] / MVA_BASE for r in 1:y) == load_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE + (enable_demand_flexibility ? demand_flex_amount[y, s, e, t] : 0) - load_shedding_extreme[y, s, e, t] + bus_power_injection[y, s, e, t] + (enable_battery_storage ? (battery_charge[y, s, e, t] - battery_discharge[y, s, e, t]) : 0)) 
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices[1:num_scenarios-num_extreme_scenarios], t = time_step_indices],  load_shedding[y, s, e, t] <= load_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE + (enable_demand_flexibility ? demand_flex_amount[y, s, e, t] : 0))
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices[num_scenarios-num_extreme_scenarios+1:end], t = time_step_indices], load_shedding_extreme[y, s, e, t] <= load_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE + (enable_demand_flexibility ? demand_flex_amount[y, s, e, t] : 0))
    end

    @expression(model, cumulative_new_wind_capacity[s=bus_indices,y=epoch_indices], sum(new_wind_capacity[s, r] for r in 1:y))
    @expression(model, cumulative_new_pv_capacity[s=bus_indices,y=epoch_indices], sum(new_pv_capacity[s, r] for r in 1:y))

    if enable_demand_flexibility
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices], sum(demand_flex_amount[y, s, e, t] for t in time_step_indices) == 0)
    end
    @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices],  (1/max_curtailment_factor) * renewable_curtailment[y, s, e, t] <= (wind_power_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE) + sum(new_wind_capacity[s, r] * wind_node_new_normalized[epoch_to_end_year[y], s, e, t] / MVA_BASE + new_pv_capacity[s, r] * pv_node_new_normalized[epoch_to_end_year[y], s, e, t] / MVA_BASE for r in 1:y))

    if enable_reserve_constraints
        for y in epoch_indices, e in scenario_indices, t in time_step_indices
            @constraint(model, sum(generator_reserve[y, i, e, t] for i in generator_indices) + sum(new_ng_cc_ccs_reserve[y, s, e, t] for s in bus_indices) + sum(new_ng_ct_reserve[y, s, e, t] for s in bus_indices) + (enable_battery_storage ? sum(battery_reserve[y, s, e, t] for s in bus_indices; init = 0) : 0) >= 0.03 * sum(load_at_node[epoch_to_end_year[y], :, e, t] / MVA_BASE) + 0.05 * sum(wind_power_at_node[epoch_to_end_year[y], :, e, t] / MVA_BASE) + 0.05 * sum(new_wind_capacity[s, r] * wind_node_new_normalized[epoch_to_end_year[y], s, e, t] / MVA_BASE + new_pv_capacity[s, r] * pv_node_new_normalized[epoch_to_end_year[y], s, e, t] / MVA_BASE for r in 1:y, s in bus_indices))
        end
    else
        @constraint(model, [y = epoch_indices, i = generator_indices, e = scenario_indices, t = time_step_indices], generator_reserve[y, i, e, t] == 0)
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], battery_reserve[y, s, e, t] == 0)
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], new_ng_cc_ccs_reserve[y, s, e, t]== 0)
        @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], new_ng_ct_reserve[y, s, e, t] == 0)
    end

    rps_targets = Dict("ME" => (2, 0.80), "NH" => (1, 0.252), "VT" => (2, 0.75),"MA" => (2, 0.35),"CT" => (2, 0.48),"RI" => (3, 0.385))
    state_to_zone_map = Dict("ME" => [1], "NH" => [2], "VT" => [3],"MA" => [4, 5, 8],"CT" => [6],"RI" => [7])

    if enable_rps_constraints
        for state in keys(rps_targets)
                @constraint(model, [y = rps_targets[state][1]:epoch_indices[end], zones=state_to_zone_map[state]], sum(sum((enable_offshore_wind ? sum(new_dc_power_flow_osw[y, _l, e, t] for _l ∈ intersect(new_line_indices, buses[s].incoming_lines); init = 0) : 0) + wind_power_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE + sum(new_wind_capacity[s, r] * wind_node_new_normalized[epoch_to_end_year[y], s, e, t] / MVA_BASE + new_pv_capacity[s, r] * pv_node_new_normalized[epoch_to_end_year[y], s, e, t] / MVA_BASE for r in 1:y) - renewable_curtailment[y, s, e, t] for e in scenario_indices, t in time_step_indices) for s in zones) + (enable_rps_noncompliance_penalty ? sum(rps_noncompliance[y, s] for s in zones) : 0) >= rps_targets[state][2] * sum(sum(load_at_node[epoch_to_end_year[y], s, e, t] / MVA_BASE + (enable_demand_flexibility ? demand_flex_amount[y, s, e, t] : 0) for e in scenario_indices, t in time_step_indices) for s in zones))
        end
    end

    @constraint(model, [y = epoch_indices, l = existing_line_indices, e = scenario_indices, t = time_step_indices], dc_power_flow[y, l, e, t] * lines[l].reactance == (voltage_angle[y, lines[l].from_bus, e, t] - voltage_angle[y, lines[l].to_bus, e, t]))
    @constraint(model, [y = epoch_indices, l = existing_line_indices, e = scenario_indices, t = time_step_indices], -lines[l].max_apparent_power / MVA_BASE <= dc_power_flow[y, l, e, t] <= lines[l].max_apparent_power / MVA_BASE)
    @constraint(model, [y = epoch_indices, s = bus_indices, e = scenario_indices, t = time_step_indices], -π <= voltage_angle[y, s, e, t] <= π)
    @constraint(model, [y = epoch_indices, s = slack_bus_idx, e = scenario_indices, t = time_step_indices], voltage_angle[y, s, e, t] == 0)
    for s in bus_indices
        if buses[s].incoming_lines == [] && buses[s].outgoing_lines == []
            @constraint(model, [y = epoch_indices, e = scenario_indices, t = time_step_indices], bus_power_injection[y, s, e, t] == 0)
        elseif buses[s].incoming_lines != [] && buses[s].outgoing_lines == []
             @constraint(model, [y = epoch_indices, e = scenario_indices, t = time_step_indices], bus_power_injection[y, s, e, t] == sum(-dc_power_flow[y, k, e, t] for k in intersect(buses[s].incoming_lines, existing_line_indices); init=0) + sum(-new_dc_power_flow[y, k, e, t] for k in intersect(buses[s].incoming_lines, upgradeable_line_indices); init=0) + (enable_offshore_wind ? sum(-new_dc_power_flow_osw[y, k, e, t] for k in intersect(buses[s].incoming_lines, new_line_indices); init=0) : 0))
        elseif buses[s].incoming_lines == [] && buses[s].outgoing_lines != []
            @constraint(model, [y = epoch_indices, e = scenario_indices, t = time_step_indices], bus_power_injection[y, s, e, t] == sum(dc_power_flow[y, k, e, t] for k in intersect(buses[s].outgoing_lines, existing_line_indices); init=0) + sum(new_dc_power_flow[y, k, e, t] for k in intersect(buses[s].outgoing_lines, upgradeable_line_indices); init=0) + (enable_offshore_wind ? sum(new_dc_power_flow_osw[y, k, e, t] for k in intersect(buses[s].outgoing_lines, new_line_indices); init=0) : 0))
        elseif buses[s].incoming_lines != [] && buses[s].outgoing_lines != []
            @constraint(model, [y = epoch_indices, e = scenario_indices, t = time_step_indices], bus_power_injection[y, s, e, t] == sum(-dc_power_flow[y, k, e, t] for k in intersect(buses[s].incoming_lines, existing_line_indices); init=0) + sum(dc_power_flow[y, k, e, t] for k in intersect(buses[s].outgoing_lines, existing_line_indices); init=0) + sum(-new_dc_power_flow[y, k, e, t] for k in intersect(buses[s].incoming_lines, upgradeable_line_indices); init=0) + sum(new_dc_power_flow[y, k, e, t] for k in intersect(buses[s].outgoing_lines, upgradeable_line_indices); init=0) + (enable_offshore_wind ? (sum(-new_dc_power_flow_osw[y, k, e, t] for k in intersect(buses[s].incoming_lines, new_line_indices); init=0) + sum(new_dc_power_flow_osw[y, k, e, t] for k in intersect(buses[s].outgoing_lines, new_line_indices); init=0)) : 0))
        end
    end

    @constraint(model, [l = upgradeable_line_indices], 0 <= sum(line_build_start[l, r] for r in epoch_indices) <= 1)
    @constraint(model,[l = upgradeable_line_indices, y = epoch_indices], line_in_service[l, y] == sum(line_build_start[l, r] for r in 1:y))
    @constraint(model,[y = epoch_indices[2:end], l = upgradeable_line_indices], line_in_service[l, y] >= line_in_service[l, y-1])

    @constraint(model, [y = epoch_indices, l = upgradeable_line_indices, e = scenario_indices, t = time_step_indices], - (π / 2 * 1 / lines[l].reactance * (1 - line_in_service[l, y])) <= new_dc_power_flow[y, l, e, t] * lines[l].reactance - (voltage_angle[y, lines[l].from_bus, e, t] - voltage_angle[y, lines[l].to_bus, e, t]))
    @constraint(model, [y = epoch_indices, l = upgradeable_line_indices, e = scenario_indices, t = time_step_indices],  new_dc_power_flow[y, l, e, t] * lines[l].reactance - (voltage_angle[y, lines[l].from_bus, e, t] - voltage_angle[y, lines[l].to_bus, e, t]) <= (π / 2 * 1 / lines[l].reactance * (1 - line_in_service[l, y])))
    @constraint(model, [y = epoch_indices, l = upgradeable_line_indices, e = scenario_indices, t = time_step_indices], -lines[l].max_apparent_power / MVA_BASE * line_in_service[l, y] <= new_dc_power_flow[y, l, e, t])
    @constraint(model, [y = epoch_indices, l = upgradeable_line_indices, e = scenario_indices, t = time_step_indices], new_dc_power_flow[y, l, e, t] <= lines[l].max_apparent_power / MVA_BASE * line_in_service[l, y])

    if enable_offshore_wind
        @constraint(model, [l = new_line_indices], sum(osw_line_build_start[l, c, r] for r in epoch_indices, c in cable_type_indices) <= 1)
        @constraint(model,[l = new_line_indices, y = epoch_indices, c = cable_type_indices], osw_line_in_service[l, c, y] == sum(osw_line_build_start[l, c, r] for r in 1:y))
        @constraint(model,[y = epoch_indices[2:end],c = cable_type_indices, l = new_line_indices], osw_line_in_service[l, c, y] >= osw_line_in_service[l, c, y-1])

        @constraint(model, [y = epoch_indices, l = new_line_indices, c = cable_type_indices[1], e = scenario_indices, t = time_step_indices], - (π / 2 * 1 / lines[l].reactance * (1 - osw_line_in_service[l, 1, y])) <= new_dc_power_flow_osw[y, l, e, t] * lines[l].reactance - (voltage_angle[y, lines[l].from_bus, e, t] - voltage_angle[y, lines[l].to_bus, e, t]))
        @constraint(model, [y = epoch_indices, l = new_line_indices, c = cable_type_indices[1], e = scenario_indices, t = time_step_indices],  new_dc_power_flow_osw[y, l, e, t] * lines[l].reactance - (voltage_angle[y, lines[l].from_bus, e, t] - voltage_angle[y, lines[l].to_bus, e, t]) <= (π / 2 * 1 / lines[l].reactance * (1 - osw_line_in_service[l, 1, y])))
        @constraint(model, [y = epoch_indices, l = new_line_indices, e = scenario_indices, t = time_step_indices], - (400 / MVA_BASE * osw_line_in_service[l, 1, y] + 1400 / MVA_BASE * osw_line_in_service[l, 2, y] + 2200 / MVA_BASE * osw_line_in_service[l, 3, y]) <= new_dc_power_flow_osw[y, l, e, t])
        @constraint(model, [y = epoch_indices, l = new_line_indices, e = scenario_indices, t = time_step_indices], new_dc_power_flow_osw[y, l, e, t] <= (400 / MVA_BASE * osw_line_in_service[l, 1, y] + 1400 / MVA_BASE * osw_line_in_service[l, 2, y] + 2200 / MVA_BASE * osw_line_in_service[l, 3, y]))
        
        for s in new_bus_indices
            if ~isempty(intersect(union(buses[s].incoming_lines, buses[s].outgoing_lines), new_line_indices))
                    @constraint(model, sum(osw_line_build_start[k, c, wind_online_year_map[s]] for k in intersect(union(buses[s].incoming_lines, buses[s].outgoing_lines), new_line_indices), c in cable_type_indices) >= 1)
            end
        end
    end
    return model
end