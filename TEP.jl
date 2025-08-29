# Set up environment from .toml files
# add DataFrames, CSV, JuMP, Gurobi, Plots, IJulia, Conda
import Pkg
Pkg.activate(".")
Pkg.instantiate()

# Load necessary packages
using LinearAlgebra, DataFrames, CSV, JuMP, Gurobi, Plots

# Load functions and model
include(joinpath(@__DIR__, "source", "input.jl")) # Type definitions and read-in functions
include(joinpath(@__DIR__, "source", "model.jl")) # Model definition 
include(joinpath(@__DIR__, "source", "output.jl")) # Postprocessing of solved model 

function diagnose_infeasibility(model)
    if termination_status(model) == MOI.INFEASIBLE
        println("Model is infeasible")

        # Access the Gurobi model
        gurobi_model_ref = model.moi_backend.optimizer.model

        # Call Gurobi's computeIIS function
        Gurobi.GRBcomputeIIS(gurobi_model_ref)

        # Write the IIS to a file
        Gurobi.GRBwrite(gurobi_model_ref, "report_infeasibility.ilp")
        println("IIS written to report_infeasibility.ilp")

        # Export the model to MPS format
        # Gurobi.GRBwrite(gurobi_model_ref, "model.mps")
        # println("Model exported to model.mps")
    end
end

case_scenarios = CSV.read(joinpath(@__DIR__, "cases.csv"), DataFrame, header=true, types=String)
for scenario_name in names(case_scenarios)[5:end]
    @info "################$(repeat("#", length(scenario_name)))################"
    @info "## RUNNING SCENARIO: \"$scenario_name\" ... ###"
    @info "################$(repeat("#", length(scenario_name)))################"
    # Scenario name for output folder
    global output_directory = isempty(scenario_name) ? joinpath(@__DIR__, "outputs","base") : joinpath(@__DIR__, "outputs",scenario_name)
    if !ispath(output_directory)
        mkpath(output_directory; mode = 0o777)
    end
    # Formulation switches and settings
    case_scenarios[!,scenario_name] .= coalesce.(case_scenarios[!,scenario_name], case_scenarios[!,"Default Value"])
    local scenario_parameters = Dict(row[Symbol("Index")] => row[(scenario_name)] for row in eachrow(case_scenarios))
    # Write original "cases.csv" file the dictionary to a CSV file
    CSV.write(joinpath(output_directory, "case.csv"), case_scenarios)
    CSV.write(joinpath(output_directory, "scenario_parameters_" * scenario_name * ".csv"), pairs(scenario_parameters), header=["Keys", "Values"])
    
    global data_directory = abspath(joinpath(String(scenario_parameters["datadir"])))
    global mip_optimality_gap = parse(Float64, String(scenario_parameters["MIPGap"]))
    global objective_scaling_factor = parse(Float64, String(scenario_parameters["ObjScale"]))
    global enable_annual_resolution = parse(Bool, String(scenario_parameters["GSw_AnnualResolution"]))
    global enable_rps_constraints = parse(Bool, String(scenario_parameters["GSw_RPS"]))
    global enable_multi_objective = parse(Bool, String(scenario_parameters["GSw_MultiObj"]))
    global enable_emission_costs = parse(Bool, String(scenario_parameters["GSw_Emission"]))
    global co2_price = parse(Float64, String(scenario_parameters["P_CO2"]))
    global enable_air_quality_costs = parse(Bool, String(scenario_parameters["GSw_AirQuality"]))
    global air_quality_cost_scaler = parse(Float64, String(scenario_parameters["k_scale_AQ"]))
    global enable_demand_flexibility = parse(Bool, String(scenario_parameters["GSw_DemandFlexibility"]))
    global demand_flexibility_factor = parse(Float64, String(scenario_parameters["alpha_flex"]))
    global enable_battery_storage = parse(Bool, String(scenario_parameters["GSw_Battery"]))
    global enable_offshore_battery = parse(Bool, String(scenario_parameters["GSw_BatteryOSW"]))
    global enable_offshore_wind = parse(Bool, String(scenario_parameters["GSw_OSW"]))
    global num_existing_lines = parse(Int, String(scenario_parameters["n_lines_existing"]))
    global battery_charging_efficiency = parse(Float64, String(scenario_parameters["eff_charging"]))
    global battery_discharging_efficiency = parse(Float64, String(scenario_parameters["eff_discharging"]))
    global calendar_degradation_rate = parse(Float64, String(scenario_parameters["k_cal"]))
    global cycle_degradation_rate = parse(Float64, String(scenario_parameters["k_cycle"]))
    global battery_duration_hours = parse(Float64, String(scenario_parameters["battery_size_hr"]))
    global enable_wind_ptc = parse(Bool, String(scenario_parameters["GSw_WindPTC"]))
    global enable_pv_itc = parse(Bool, String(scenario_parameters["GSw_PVITC"]))
    global enable_exogenous_retirements = parse(Bool, String(scenario_parameters["GSw_ExogenousRetirements"]))
    global onshore_line_cost_scaler = parse(Float64, String(scenario_parameters["k_scale_OnshoreLineCost"]))
    global enable_rps_noncompliance_penalty = parse(Bool, String(scenario_parameters["GSw_RPSNC"]))
    global rps_noncompliance_penalty_cost = parse(Float64, String(scenario_parameters["RPSNC"]))
    global enable_full_uc = parse(Bool, String(scenario_parameters["GSw_FullUC"]))
    global value_of_lost_load = parse(Float64, String(scenario_parameters["VoLL"]))
    global value_of_wind_spillage = parse(Float64, String(scenario_parameters["VoWS"]))
    global value_of_offshore_wind_spillage = parse(Float64, String(scenario_parameters["VoOSWS"]))
    global discount_rate = parse(Float64, String(scenario_parameters["r"]))
    global time_step_duration = parse(Float64, String(scenario_parameters["delta_t"]))
    global transmission_lifetime = parse(Int, String(scenario_parameters["LT"]))
    global generator_lifetime = parse(Int, String(scenario_parameters["GT"]))
    global storage_lifetime = parse(Int, String(scenario_parameters["ST"]))
    global num_existing_buses = parse(Int, String(scenario_parameters["n_buses_existing"]))
    global num_planning_years = parse(Int, String(scenario_parameters["Y"]))
    global planning_start_year = parse(Int, String(scenario_parameters["Y_start"]))
    global num_scenarios = parse(Int, String(scenario_parameters["e_num"]))
    global enable_extreme_scenarios = parse(Bool, String(scenario_parameters["GSw_ExtremeScenarios"]))
    global num_extreme_scenarios = parse(Int, String(scenario_parameters["e_numx"]))
    global value_of_lost_load_extreme = parse(Float64, String(scenario_parameters["VoLLX"]))
    global extreme_scenario_weight = parse(Float64, String(scenario_parameters["alpha_E"]))
    global max_curtailment_factor = parse(Float64, String(scenario_parameters["alpha_plus"]))

    # Load data
    local power_grid = load_network(data_directory)
    local wind_power_profile = Matrix(CSV.read(joinpath(data_directory, "wind.csv"), DataFrame, header=false))'
    local pv_power_profile = Matrix(CSV.read(joinpath(data_directory, "solar.csv"), DataFrame, header=false))'
    local load_profile = Matrix(CSV.read(joinpath(data_directory, "load.csv"), DataFrame, header=true, transpose=true))[:, 2:end]
    local load_node_count, total_time_steps = size(load_profile)
    local wind_node_count, total_time_steps = size(wind_power_profile)
    local pv_node_count, total_time_steps = size(pv_power_profile)
    total_time_steps = parse(Int, String(scenario_parameters["t_num"]))

    if enable_extreme_scenarios
        num_scenarios = num_scenarios + num_extreme_scenarios
        println("num_scenarios: $num_scenarios")
        println("num_extreme_scenarios: $num_extreme_scenarios")
    end

    local optimization_model = build_model(power_grid, wind_power_profile, pv_power_profile, load_profile, wind_node_count, pv_node_count, total_time_steps)
    local solve_duration = @elapsed optimize!(optimization_model)
    @show termination_status(optimization_model)
    diagnose_infeasibility(optimization_model)
    @show solution_summary(optimization_model; verbose = false)
    @show raw_status(optimization_model)
    @show objective_value(optimization_model)    
    save_results(optimization_model, power_grid, total_time_steps)

end

