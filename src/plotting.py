import matplotlib.pyplot as plt
import numpy as np

### PLOTTING PREDICTIONS
def plot_data_q(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['DXL_Position'] * 180/np.pi, label='real')
    plt.plot(df['q_pred'] * 180/np.pi, label='pred')
    plt.xlabel('Time (sample)')
    plt.ylabel('Values')
    plt.title(f'Position real vs pred')
    plt.legend()
    plt.show()


def plot_data_q_dot(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['DXL_Velocity'] * 180/np.pi, label='real')
    plt.plot(df['q_dot_pred'] * 180/np.pi, label='pred')
    plt.xlabel('Time (sample)')
    plt.ylabel('Values')
    plt.title(f'Velocity real vs pred')
    plt.legend()
    plt.show()


def plot_torques(df, file_name):
    plt.figure(figsize=(10, 6))
    plt.plot(df['tau_f'], label='tau_f')
    # plt.plot(df['tau_I'], label='tau_I')
    # plt.plot(df['tau_U'], label='tau_U')
    # plt.plot(df['tau_o'], label='tau_o')
    # plt.plot(df['tau_datasheet'], label='tau_datasheet')
    plt.plot(df['tau_em'], label='tau_em')
    plt.xlabel('Time (sample)')
    plt.ylabel('Torque [Nm]')
    plt.title(f'Torques {file_name}')
    plt.legend()
    plt.show()


def plot_computed_acceleration(df, file_name):
    plt.figure(figsize=(10, 6))
    plt.plot(df['a'], label='acceleration')
    plt.xlabel('Time (sample)')
    plt.ylabel('Acceleration [rad/s^2]')
    plt.title(f'Acceleration {file_name}')
    plt.legend()
    plt.show()



### PLOTTING MEASURES ['DXL_PWM', 'DXL_Current', 'DXL_Velocity', 'DXL_Position', 'DXL_Input_Voltage']
    
def plot_measured_I(df_slow, df_fast=None, label1='Experiment 1', label2='Experiment 2'):
    plt.figure(figsize=(10, 6))
    plt.plot(df_slow['t'], df_slow['DXL_Current'] * 1000, label=label1)
    if df_fast is not None :
        plt.plot(df_fast['t'], df_fast['DXL_Current'] * 1000, label=label2)
        plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel("Current \n [mA]", rotation=0, labelpad=40)
    #plt.title(f'Measured current')
    plt.show()

def plot_measured_U(df_slow, df_fast=None, label1='Experiment 1', label2='Experiment 2'):
    plt.figure(figsize=(10, 6))
    plt.plot(df_slow['t'], df_slow['U'], label=label1)
    if df_fast is not None :
        plt.plot(df_fast['t'], df_fast['U'], label=label2)
        plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel("Voltage \n [V]", rotation=0, labelpad=40)
    # plt.grid()
    #Òplt.title(f'Measured Voltage')
    plt.show()
    
def plot_measured_q(df_slow, df_fast=None, label1='Experiment 1', label2='Experiment 2'):
    plt.figure(figsize=(10, 6))
    plt.plot(df_slow['t'], df_slow['DXL_Position'], label=label1)
    # plt.plot(df_slow['DXL_Position'] * 180/np.pi, label=label1)

    if df_fast is not None :
        plt.plot(df_fast['DXL_Position'] * 180/np.pi, label=label2)
        plt.legend()
    plt.xlabel('Time (sample)')
    plt.ylabel('Position [°]')
    plt.title(f'Measured Position')
    plt.grid()
    plt.show()

def plot_measured_q_dot(df_slow, df_fast=None, label1='Values 1', label2='Values 2'):
    plt.figure(figsize=(10, 6))
    plt.plot(df_slow['DXL_Velocity'], label=label1)
    if df_fast is not None :
        plt.plot(df_fast['DXL_Velocity'], label=label2)
        plt.legend()
    plt.xlabel('Time (sample)')
    plt.ylabel('Velocity [rad/s]')
    plt.title(f'Measured Velocity')
    plt.show()
    
def plot_tau_o(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['tau_o'])
    plt.xlabel('Time (sample)')
    plt.ylabel('Torque [Nm]')
    plt.title(f'Estimated effective torque')
    plt.show()

def plot_tau_I(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['tau_I'])
    plt.xlabel('Time (sample)')
    plt.ylabel('Torque [Nm]')
    plt.title(f'Estimated electromagnetic torque using Current')
    plt.show()

def plot_tau_U(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['tau_U'])
    plt.xlabel('Time (sample)')
    plt.ylabel('Torque [Nm]')
    plt.title(f'Estimated electromagnetic torque using Voltage')
    plt.show()

def plot_tau_f(df, source='tau_f'):
    plt.figure(figsize=(10, 6))
    plt.plot(df[source])
    plt.xlabel('Time (sample)')
    plt.ylabel('Torque [Nm]')
    plt.title(f'Estimated friction torque')
    plt.show()

def plot_accelerations(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['a'], label='a')
    plt.plot(df['Acceleration_from_Velocity_from_position'], label='Acceleration_from_Velocity_from_position')
    plt.plot(df['Acceleration_from_Velocity_from_position_filtered'], label='Acceleration_from_Velocity_from_position_filtered')
    plt.plot(df['Acceleration_from_DXL_Velocity'], label='Acceleration_from_DXL_Velocity')
    plt.plot(df['Acceleration_from_DXL_Velocity_filtered'], label='Acceleration_from_DXL_Velocity_filtered')
    plt.xlabel('Time (sample)')
    plt.ylabel('Acceleration [rad/s^2]')
    # plt.title(f'Estimated effective torque')
    plt.legend()
    plt.show()


### Some scatter plots
def scatter_measures(df, x_axis, y_axis, x_label, y_label):
    grouped = df.groupby([x_axis, y_axis]).size().reset_index(name='counts')
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(grouped[y_axis], grouped[x_axis], c=grouped['counts'], cmap='viridis', alpha=0.6, edgecolors='w', s=50)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.ylabel('kphi value [V.s/rad]')
    cb = plt.colorbar(sc)
    cb.set_label('Number of Recurrences')
    # Adding the legend
    plt.legend()
    plt.show()
