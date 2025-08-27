mutable struct ThermalGenerator
  index::Int
  plant_type::String
  location_node::Int
  active_power_output::Float64
  reactive_power_output::Float64
  max_reactive_power::Float64
  min_reactive_power::Float64
  voltage_magnitude::Float64
  mva_base::Float64
  status::Int
  max_power_output::Float64
  min_power_output::Float64
  cost_coeff_quadratic::Float64
  cost_coeff_linear::Float64
  cost_coeff_constant::Float64
  startup_cost::Float64
  shutdown_cost::Float64
  ramp_up_rate::Float64
  ramp_down_rate::Float64
  min_uptime::Float64
  min_downtime::Float64
  zone::String
  avg_marginal_damage::Float64
  co2_tons_per_mwh::Float64
  so2_tons_per_mwh::Float64
  nox_tons_per_mwh::Float64
  pm25_tons_per_mwh::Float64
  planned_retirement_year::Int
  function ThermalGenerator(index, plant_type, location_node, active_power_output, reactive_power_output, max_reactive_power, min_reactive_power, voltage_magnitude, mva_base, status, max_power_output, min_power_output, cost_coeff_quadratic, cost_coeff_linear, cost_coeff_constant, startup_cost, shutdown_cost, ramp_up_rate, ramp_down_rate, min_uptime, min_downtime, zone, avg_marginal_damage, co2_tons_per_mwh, so2_tons_per_mwh, nox_tons_per_mwh, pm25_tons_per_mwh, planned_retirement_year)
     gen = new()
     gen.index = index
     gen.plant_type = plant_type
     gen.location_node = location_node
     gen.active_power_output = active_power_output
     gen.reactive_power_output = reactive_power_output
     gen.max_reactive_power = max_reactive_power
     gen.min_reactive_power = min_reactive_power
     gen.voltage_magnitude = voltage_magnitude
     gen.mva_base = mva_base
     gen.status = status
     gen.max_power_output = max_power_output
     gen.min_power_output = min_power_output
     gen.cost_coeff_quadratic = cost_coeff_quadratic
     gen.cost_coeff_linear = cost_coeff_linear
     gen.cost_coeff_constant = cost_coeff_constant
     gen.startup_cost = startup_cost
     gen.shutdown_cost = shutdown_cost
     gen.ramp_up_rate = ramp_up_rate
     gen.ramp_down_rate = ramp_down_rate
     gen.min_uptime = min_uptime
     gen.min_downtime = min_downtime
     gen.zone = zone
     gen.avg_marginal_damage = avg_marginal_damage
     gen.co2_tons_per_mwh = co2_tons_per_mwh
     gen.so2_tons_per_mwh = so2_tons_per_mwh
     gen.nox_tons_per_mwh = nox_tons_per_mwh
     gen.pm25_tons_per_mwh = pm25_tons_per_mwh
     gen.planned_retirement_year = planned_retirement_year
     return gen
  end
end

mutable struct Bus
  index::Int
  node_id::Int
  zone::String
  is_slack::Bool
  active_power_demand::Float64
  reactive_power_demand::Float64
  max_voltage::Float64
  min_voltage::Float64
  shunt_conductance::Float64
  shunt_susceptance::Float64
  base_kv::Float64  
  incoming_lines::Vector{Int}
  outgoing_lines::Vector{Int}
  generators_at_bus::Vector{Int}
  function Bus(index, node_id, zone, is_slack, active_power_demand, reactive_power_demand, max_voltage, min_voltage, shunt_conductance, shunt_susceptance, base_kv)
     bus = new()
     bus.index = index
     bus.node_id = node_id
     bus.zone = zone
     bus.is_slack = is_slack
     bus.active_power_demand = active_power_demand
     bus.reactive_power_demand = reactive_power_demand
     bus.max_voltage = max_voltage
     bus.min_voltage = min_voltage
     bus.shunt_conductance = shunt_conductance
     bus.shunt_susceptance = shunt_susceptance
     bus.base_kv = base_kv
     bus.incoming_lines = Int[]
     bus.outgoing_lines = Int[]
     bus.generators_at_bus = Int[]
     return bus
  end
end

mutable struct Line
  index::Int
  from_bus::Int
  to_bus::Int
  resistance::Float64
  reactance::Float64
  susceptance::Float64
  max_apparent_power::Float64
  function Line(index, from_bus, to_bus, resistance, reactance, susceptance, max_apparent_power)
     line = new()
     line.index = index
     line.from_bus = from_bus
     line.to_bus = to_bus
     line.resistance = resistance
     line.reactance = reactance
     line.susceptance = susceptance
     line.max_apparent_power = max_apparent_power
     return line
  end
end

mutable struct PowerSystem
 buses::Array{Bus}
 lines::Array{Line}
 generators::Array{ThermalGenerator}
 num_buses::Int
 num_lines::Int
 num_generators::Int
 slack_bus_index::Int
 function PowerSystem(buses, lines, generators)
   net = new()
   net.buses = buses
   net.lines = lines
   net.generators = generators
   net.num_buses = length(buses)
   net.num_lines = length(lines)
   net.num_generators = length(generators)
   for (i, bus) in enumerate(buses)
     if bus.is_slack
       net.slack_bus_index = i
       break
     end
   end
   return net
 end
end

function load_network(data_path)
# READ RAW DATA
println("Reading raw data from $(data_path)")

 raw_nodes_data = CSV.read(joinpath(data_path, "nodes.csv"), DataFrame)
 sum(nonunique(raw_nodes_data, :index)) != 0 ? @warn("Ambiguous Node Indices") : nothing

 bus_collection = []
 for n in 1:nrow(raw_nodes_data)
     index = raw_nodes_data[n, :index]
     node_id = raw_nodes_data[n, :node]
     zone = raw_nodes_data[n, :Zone]
     is_slack = raw_nodes_data[n, :is_slack]
     active_power_demand = raw_nodes_data[n, :Pd]
     reactive_power_demand = raw_nodes_data[n, :Qd]
     max_voltage = raw_nodes_data[n, :Vmax]
     min_voltage = raw_nodes_data[n, :Vmin]
     shunt_conductance = raw_nodes_data[n, :Gs]
     shunt_susceptance = raw_nodes_data[n, :Bs]
     base_kv = raw_nodes_data[n, :baseKV]
     new_bus = Bus(index, node_id, zone, is_slack, active_power_demand, reactive_power_demand, max_voltage, min_voltage, shunt_conductance, shunt_susceptance, base_kv)
     push!(bus_collection, new_bus)
 end

 raw_generators_data = CSV.read(joinpath(data_path,"gen_loc_emission_damage.csv"), DataFrame)
 sum(nonunique(raw_generators_data, :index)) != 0 ? @warn("Ambiguous Generator Indices") : nothing

 raw_generators_data[:,:Pg] .= 0
 raw_generators_data[:,:Qg] .= 0
 raw_generators_data[:,:Qmax] .= 0
 raw_generators_data[:,:Qmin] .= 0
 raw_generators_data[:,:Vg] .= 0
 raw_generators_data[:,:mBase] .= 1
 raw_generators_data[:,:status] .= 0

 generator_collection = []
 for g in 1:nrow(raw_generators_data)
     index = raw_generators_data[g, :index]
     plant_type = raw_generators_data[g, :type]
     location_node = raw_generators_data[g, :location_node]
     active_power_output = raw_generators_data[g, :Pg]
     reactive_power_output = raw_generators_data[g, :Qg]
     max_reactive_power = raw_generators_data[g, :Qmax]
     min_reactive_power = raw_generators_data[g, :Qmin]
     voltage_magnitude = raw_generators_data[g, :Vg]
     mva_base = raw_generators_data[g, :mBase]
     status = raw_generators_data[g, :status]
     max_power_output = raw_generators_data[g, :Pmax]
     min_power_output = raw_generators_data[g, :Pmin]
     cost_coeff_quadratic = raw_generators_data[g, :c2]
     cost_coeff_linear = raw_generators_data[g, :c1]
     cost_coeff_constant = raw_generators_data[g, :c0]
     startup_cost = raw_generators_data[g, :SUcost]
     shutdown_cost = raw_generators_data[g, :SDcost]
     ramp_up_rate = raw_generators_data[g, :RUrate]
     ramp_down_rate = raw_generators_data[g, :RDrate]
     min_uptime = raw_generators_data[g, :UPtime]
     min_downtime = raw_generators_data[g, :DNtime]
     zone = raw_generators_data[g, :Zone]
     avg_marginal_damage = raw_generators_data[g, :ave_marg_damages_isrm_LePeule]
     co2_tons_per_mwh = raw_generators_data[g, :CO2mtonperMWh]
     so2_tons_per_mwh = raw_generators_data[g, :SO2mtonperMWh]
     nox_tons_per_mwh = raw_generators_data[g, :NOxmtonperMWh]
     pm25_tons_per_mwh = raw_generators_data[g, :PM25mtonperMWh]
     planned_retirement_year = raw_generators_data[g, :PlannedRetirementYear]
     new_gen = ThermalGenerator(index, plant_type, location_node, active_power_output, reactive_power_output, max_reactive_power, min_reactive_power, voltage_magnitude, mva_base, status, max_power_output, min_power_output, cost_coeff_quadratic, cost_coeff_linear, cost_coeff_constant, startup_cost, shutdown_cost, ramp_up_rate, ramp_down_rate, min_uptime, min_downtime, zone, avg_marginal_damage, co2_tons_per_mwh, so2_tons_per_mwh, nox_tons_per_mwh, pm25_tons_per_mwh, planned_retirement_year)
     for n in 1:nrow(raw_nodes_data)
        if bus_collection[n].node_id==new_gen.location_node
             push!(bus_collection[n].generators_at_bus, new_gen.index)
         end
     end
     push!(generator_collection, new_gen)
 end

 raw_lines_data = CSV.read(joinpath(data_path,"lines.csv"), DataFrame)
 sum(nonunique(raw_lines_data, :index)) != 0  ? @warn("Ambiguous Line Indices") : nothing

 line_collection = []
 for l in 1:nrow(raw_lines_data)
     index = raw_lines_data[l, :index]
     from_bus = raw_lines_data[l, :from_node]
     to_bus = raw_lines_data[l, :to_node]
     resistance = raw_lines_data[l, :r]
     reactance = raw_lines_data[l, :x]
     susceptance = raw_lines_data[l, :b]
     max_apparent_power = raw_lines_data[l, :s_max]
     new_line = Line(index, from_bus, to_bus, resistance, reactance, susceptance, max_apparent_power)
     for n in 1:nrow(raw_nodes_data)
         if bus_collection[n].node_id == new_line.from_bus
             push!(bus_collection[n].outgoing_lines, new_line.index)
         elseif bus_collection[n].node_id == new_line.to_bus
             push!(bus_collection[n].incoming_lines, new_line.index)
         end
     end
     push!(line_collection, new_line)
 end

 power_grid = PowerSystem(bus_collection, line_collection, generator_collection)

 return power_grid
end