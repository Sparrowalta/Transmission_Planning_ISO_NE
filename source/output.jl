function save_results(solved_model, grid_data, time_steps_per_scenario)
    generators = grid_data.generators
    num_generators = grid_data.num_generators
    num_buses = grid_data.num_buses
    num_lines = grid_data.num_lines

    time_step_indices = collect(1:time_steps_per_scenario)
    bus_indices = collect(1:num_buses)
    scenario_indices = collect(1:num_scenarios)

    function _reshape_line_data_to_2d(mat4d)
        mat2d = zeros(num_lines*num_scenarios, length(epoch_indices)*time_steps_per_scenario)
        for e=1:num_scenarios, l=1:num_lines
            for y=epoch_indices, t=1:time_steps_per_scenario
                 mat2d[l+num_lines*(e-1), t+time_steps_per_scenario*(y-1)] = mat4d[y, l, e, t]
            end
        end
        return mat2d
    end

    function _reshape_node_data_to_2d(mat4d)
        mat2d = zeros(num_buses*num_scenarios, length(epoch_indices)*time_steps_per_scenario)
        for e=1:num_scenarios, s=1:num_buses
            for y=epoch_indices, t=1:time_steps_per_scenario
                mat2d[s+num_buses*(e-1), t+time_steps_per_scenario*(y-1)] = mat4d[y, s, e, t]
            end
        end
        return mat2d
    end

    function _reshape_line_build_data_to_2d(mat3d)
        mat2d = zeros(num_lines, length(epoch_indices)*3)
        for l=1:num_lines
            for y=epoch_indices, c=1:3
                mat2d[l, c+3*(y-1)] = mat3d[l, c, y]
            end
        end
        return mat2d
    end

    thermal_gen_values = value.(solved_model[:thermal_generation]) * MVA_BASE
    thermal_gen_2d = zeros(num_generators*num_scenarios, length(epoch_indices)*time_steps_per_scenario)
    for e=1:num_scenarios, i=1:num_generators
        for y=epoch_indices, t=1:time_steps_per_scenario
            thermal_gen_2d[i+num_generators*(e-1), t+time_steps_per_scenario*(y-1)] = thermal_gen_values[y, i, e, t]
        end
    end
    CSV.write(joinpath(output_directory, "power_output_existing_generators.csv"), Tables.table(thermal_gen_2d * MVA_BASE), header=false)
    CSV.write(joinpath(output_directory, "power_output_new_ng_cc_ccs.csv"), Tables.table(_reshape_node_data_to_2d(value.(solved_model[:new_ng_cc_ccs_generation]) * MVA_BASE)), header=false)
    CSV.write(joinpath(output_directory, "power_output_new_ng_ct.csv"), Tables.table(_reshape_node_data_to_2d(value.(solved_model[:new_ng_ct_generation]) * MVA_BASE)), header=false)
    CSV.write(joinpath(output_directory, "load_node_out.csv"), Tables.table(_reshape_node_data_to_2d(final_load) * MVA_BASE), header=false)
    CSV.write(joinpath(output_directory, "wind_node_out.csv"), Tables.table(_reshape_node_data_to_2d(final_wind_power) * MVA_BASE), header=false)
    CSV.write(joinpath(output_directory, "pv_node_out.csv"), Tables.table(_reshape_node_data_to_2d(final_pv_power) * MVA_BASE), header=false)
    CSV.write(joinpath(output_directory, "wind_node_new_out_raw.csv"), Tables.table(_reshape_node_data_to_2d(final_new_wind_profile) * MVA_BASE), header=false)
    CSV.write(joinpath(output_directory, "wind_node_new_out.csv"), Tables.table(_reshape_node_data_to_2d(final_new_wind_profile)  * MVA_BASE .* repeat(repeat(value.(solved_model[:cumulative_new_wind_capacity])', inner=(time_steps_per_scenario,1))',num_scenarios)), header=false)
    CSV.write(joinpath(output_directory, "pv_node_new_out_raw.csv"), Tables.table(_reshape_node_data_to_2d(final_new_pv_profile) * MVA_BASE), header=false)
    CSV.write(joinpath(output_directory, "pv_node_new_out.csv"), Tables.table(_reshape_node_data_to_2d(final_new_pv_profile) * MVA_BASE .* repeat(repeat(value.(solved_model[:cumulative_new_pv_capacity])', inner=(time_steps_per_scenario,1))',num_scenarios)), header=false)

    thermal_gen_at_node_2d = zeros(num_buses*num_scenarios, length(epoch_indices)*time_steps_per_scenario)
    for e=1:num_scenarios, s=1:num_buses
        for y=epoch_indices, t=1:time_steps_per_scenario
            thermal_gen_at_node_2d[s+num_buses*(e-1), t+time_steps_per_scenario*(y-1)] = isempty(grid_data.buses[s].generators_at_bus) ? 0 : sum(thermal_gen_values[y, i, e, t] for i in grid_data.buses[s].generators_at_bus)
        end
    end
    CSV.write(joinpath(output_directory, "power_output_existing_generators_node.csv"), Tables.table(thermal_gen_at_node_2d * MVA_BASE), header=false)

    CSV.write(joinpath(output_directory, "load_curtailments_normal.csv"), Tables.table(_reshape_node_data_to_2d(value.(solved_model[:load_shedding]) * MVA_BASE)), header=false)
    CSV.write(joinpath(output_directory, "load_curtailments_extreme.csv"), Tables.table(_reshape_node_data_to_2d(value.(solved_model[:load_shedding_extreme]) * MVA_BASE)), header=false)
    CSV.write(joinpath(output_directory, "load_curtailments.csv"), Tables.table(_reshape_node_data_to_2d(value.(solved_model[:load_shedding]) * MVA_BASE .+ value.(solved_model[:load_shedding_extreme]) * MVA_BASE)), header=false)

    curtailment_values = value.(solved_model[:renewable_curtailment]) * MVA_BASE
    CSV.write(joinpath(output_directory, "wind_spillage.csv"), Tables.table(_reshape_node_data_to_2d(curtailment_values)), header=false)

    bus_injection_values = value.(solved_model[:bus_power_injection]) * MVA_BASE
    CSV.write(joinpath(output_directory, "bus_out_power.csv"), Tables.table(_reshape_node_data_to_2d(bus_injection_values)), header=false)
    
    if enable_battery_storage
        CSV.write(joinpath(output_directory, "ch_node.csv"), Tables.table(_reshape_node_data_to_2d(value.(solved_model[:battery_charge] ./ battery_charging_efficiency) * MVA_BASE)), header=false)
        CSV.write(joinpath(output_directory, "dis_node.csv"), Tables.table(_reshape_node_data_to_2d(value.(solved_model[:battery_discharge] ./ battery_discharging_efficiency) * MVA_BASE)), header=false)
        CSV.write(joinpath(output_directory, "new_build_gen_cap_bess_power.csv"), Tables.table(value.(solved_model[:bess_power_capacity]) * MVA_BASE); header=false)
        CSV.write(joinpath(output_directory, "new_build_gen_cap_bess_energy.csv"), Tables.table(value.(solved_model[:bess_energy_capacity]) * MVA_BASE); header=false)
        CSV.write(joinpath(output_directory, "annual_cost_bess_investment.csv"), Tables.table(value.(solved_model[:bess_investment_cost])); header=false)
    end
    
    dc_flow_values = value.(solved_model[:dc_power_flow]) * MVA_BASE
    CSV.write(joinpath(output_directory, "line_flow.csv"), Tables.table(_reshape_line_data_to_2d(dc_flow_values)), header=false)

    theta_values = value.(solved_model[:voltage_angle])
    CSV.write(joinpath(output_directory, "theta.csv"), Tables.table(_reshape_node_data_to_2d(theta_values)), header=false)

    new_dc_flow_values = value.(solved_model[:new_dc_power_flow]) * MVA_BASE
    CSV.write(joinpath(output_directory, "upgraded_line_flow.csv"), Tables.table(_reshape_line_data_to_2d(new_dc_flow_values)), header=false)

    total_cost_values = value.(solved_model[:annual_operating_cost]) .+ value.(solved_model[:generator_investment_cost]) .+ value.(solved_model[:line_investment_cost]) .+ (enable_emission_costs ? value.(solved_model[:co2_emission_cost]) : 0) .+ (enable_air_quality_costs ? value.(solved_model[:air_quality_damage_cost]) : 0) .+ (enable_battery_storage ? value.(solved_model[:bess_investment_cost]) : 0)
    CSV.write(joinpath(output_directory, "annual_cost_total.csv"), Tables.table(total_cost_values); header=false)
    CSV.write(joinpath(output_directory, "annual_cost_operation.csv"), Tables.table(value.(solved_model[:annual_operating_cost])); header=false)
    CSV.write(joinpath(output_directory, "annual_cost_gen_investment.csv"), Tables.table(value.(solved_model[:generator_investment_cost])); header=false)
    CSV.write(joinpath(output_directory, "annual_cost_line_investment.csv"), Tables.table(value.(solved_model[:line_investment_cost])); header=false)

    CSV.write(joinpath(output_directory, "annual_cost_operation_over_generation.csv"), Tables.table(value.(solved_model[:over_generation_penalty])); header=false)
    CSV.write(joinpath(output_directory, "annual_cost_operation_under_generation.csv"), Tables.table(value.(solved_model[:under_generation_penalty])); header=false)
    CSV.write(joinpath(output_directory, "annual_cost_operation_rps_noncompliance.csv"), Tables.table(value.(solved_model[:rps_noncompliance_penalty])); header=false)

    CSV.write(joinpath(output_directory, "new_build_gen_cap_ng_cc_ccs.csv"), Tables.table(value.(solved_model[:new_ng_cc_ccs_capacity]) * MVA_BASE); header=false)
    CSV.write(joinpath(output_directory, "new_build_gen_cap_ng_ct.csv"), Tables.table(value.(solved_model[:new_ng_ct_capacity]) * MVA_BASE); header=false)
    CSV.write(joinpath(output_directory, "new_build_gen_cap_wind.csv"), Tables.table(value.(solved_model[:new_wind_capacity]) * MVA_BASE); header=false)
    CSV.write(joinpath(output_directory, "new_build_gen_cap_pv.csv"), Tables.table(value.(solved_model[:new_pv_capacity]) * MVA_BASE); header=false)

    line_upgrade_decision_values = round.(Int, value.(solved_model[:line_build_start]))
    CSV.write(joinpath(output_directory, "line_upgrade_decision.csv"), Tables.table(line_upgrade_decision_values); header=false)
    
    if enable_demand_flexibility
        flexible_demand_values = value.(solved_model[:demand_flex_amount]) * MVA_BASE
        CSV.write(joinpath(output_directory, "flexible_load.csv"), Tables.table(_reshape_node_data_to_2d(flexible_demand_values)), header=false)
        CSV.write(joinpath(output_directory, "annual_cost_operation_demand_flexibility.csv"), Tables.table(value.(solved_model[:demand_flexibility_cost])); header=false)
    end

    CSV.write(joinpath(output_directory, "Emissions_CO2.csv"), Tables.table(value.(solved_model[:total_co2_emissions])); header=false)
    CSV.write(joinpath(output_directory, "Emissions_SO2.csv"), Tables.table(value.(solved_model[:total_so2_emissions])); header=false)
    CSV.write(joinpath(output_directory, "Emissions_NOx.csv"), Tables.table(value.(solved_model[:total_nox_emissions])); header=false)
    CSV.write(joinpath(output_directory, "Emissions_PM25.csv"), Tables.table(value.(solved_model[:total_pm25_emissions])); header=false)
    CSV.write(joinpath(output_directory, "annual_cost_externalities_emission.csv"), Tables.table(value.(solved_model[:co2_emission_cost])); header=false)
    CSV.write(joinpath(output_directory, "annual_cost_externalities_airquality.csv"), Tables.table(value.(solved_model[:air_quality_damage_cost])); header=false)
    CSV.write(joinpath(output_directory, "annual_cost_externalities.csv"), Tables.table(value.(solved_model[:co2_emission_cost]) .+ value.(solved_model[:air_quality_damage_cost])); header=false)

    if enable_offshore_wind
        CSV.write(joinpath(output_directory, "new_line_flow.csv"), Tables.table(_reshape_line_data_to_2d(value.(solved_model[:new_dc_power_flow_osw]) * MVA_BASE)), header=false)
        CSV.write(joinpath(output_directory, "new_line_decision.csv"), Tables.table(_reshape_line_build_data_to_2d(round.(Int, value.(solved_model[:osw_line_build_start])))); header=false)
    end

end