module torque_references

# Constants
ramp_up_time = 12
high_time = 3
ramp_down_time = 0
zero_time = 10
cycle_time = ramp_up_time + high_time + ramp_down_time + zero_time
values = [10, 15, 20, 25, 30, 35, 40]  # Updated values
values = [val for pair in zip(values, map(x -> -x, values)) for val in pair]

# Function to calculate current based on the current time
function current_for_time(t)
    cycle_index = floor(Int, t / cycle_time) + 1
    if cycle_index > length(values)
        return 0.0  # Beyond the defined pattern, return 0
    end
    val = values[cycle_index]
    cycle_phase_time = t % cycle_time
    
    if cycle_phase_time < ramp_up_time
        # Ramp up phase
        return (val / ramp_up_time) * cycle_phase_time
    elseif cycle_phase_time < ramp_up_time + high_time
        # High phase
        return val
    elseif cycle_phase_time < ramp_up_time + high_time + ramp_down_time
        # Ramp down phase (not used here since ramp_down_time = 0)
        return val - (val / ramp_down_time) * (cycle_phase_time - ramp_up_time - high_time)
    else
        # Zero phase
        return 0.0
    end
end

# Constants
const fundamental_period = 12  # Fundamental period for both waves
const omega = 2 * π / fundamental_period  # Angular frequency of the fundamental frequency
# const amplitude = 0.14  # Amplitude of the current waves
const amplitude = 2 # Amplitude of the voltage waves

# Function to return the value of a Fourier series-based waveform at time t
function waveform_value_at_time(t, type)
    n_harmonics = 5  # Number of harmonics to include
    
    # Helper function for square wave Fourier series
    function square_wave(t, omega, amplitude, n_harmonics)
        value = 0.0
        for n in 1:2:(n_harmonics * 2 - 1)  # Only odd harmonics
            value += (4 * amplitude / (n * π)) * sin(n * omega * t)
        end
        return value
    end
    
    # Helper function for sawtooth wave Fourier series
    function sawtooth_wave(t, omega, amplitude, n_harmonics)
        value = 0.0
        for n in 1:n_harmonics  # Both odd and even harmonics
            value += (2 * amplitude / (n * π)) * (-1)^(n + 1) * sin(n * omega * t)
        end
        return value
    end
    
    # Helper function for triangle wave Fourier series
    function triangle_wave(t, omega, amplitude, n_harmonics)
        value = 0.0
        for n in 1:2:(n_harmonics * 2 - 1)  # Only odd harmonics
            value += ((8 * amplitude) / ((n * π) ^ 2)) * sin(n * omega * t)
        end
        return value
    end
    
    # Determine which waveform to generate based on the 'type' parameter
    if type == "square"
        return square_wave(t, omega, amplitude, n_harmonics)
    elseif type == "sawtooth"
        return sawtooth_wave(t, omega, amplitude, n_harmonics)
    elseif type == "triangle"
        return triangle_wave(t, omega, amplitude * 1.5, n_harmonics)  # Adjust amplitude if needed
    else
        return 0.0  # Default case, should not happen if 'type' is correctly specified
    end
end

end