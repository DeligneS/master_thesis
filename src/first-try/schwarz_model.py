import numpy as np


def schwarz_model_fail(df):
    parameters_scwharz = [0.19817, 0.49586, 0.13729, 0.03006]
    alpha_0, alpha_1, alpha_2, alpha_3 = parameters_scwharz
    # Si on veut un peu plus de réalisme, on peut retrouver le k_gear du MX-106 et le changer dans alpha_2 X k_gear de notre motor
    q_dot_s = 0.1
    ß = np.exp(-abs(df['rot_speed']/q_dot_s)) # assume ∂ = 1
    df['tau_o'] = (df['U'] - alpha_1 * df['rot_speed'] - alpha_2 * np.sign(df['rot_speed']) * (1 - ß) - alpha_3 * np.sign(df['rot_speed']) * ß)/alpha_0
    # ß = np.exp(-abs(df['DXL_Velocity']/q_dot_s)) # assume ∂ = 1
    # df['tau_o'] = (df['U'] - alpha_1 * df['DXL_Velocity'] - alpha_2 * np.sign(df['DXL_Velocity']) * (1 - ß) - alpha_3 * np.sign(df['DXL_Velocity']) * ß)/alpha_0
    # -> en fait c'est faux cette écriture, les paramètres qui sont donnés dans le papier c'est pour le position crtl



def schwarz_model(df):
    parameters_scwharz = [0.19817, 0.49586, 0.13729, 0.03006]
    alpha_0, alpha_1, alpha_2, alpha_3 = parameters_scwharz
    # Si on veut un peu plus de réalisme, on peut retrouver le k_gear du MX-106 et le changer dans alpha_2 X k_gear de notre motor
    q_dot_s = 0.1
    ß = np.exp(-abs(df['rot_speed']/q_dot_s)) # assume ∂ = 1
    df['tau_o'] = (df['U'] - alpha_1 * df['rot_speed'] - alpha_2 * np.sign(df['rot_speed']) * (1 - ß) - alpha_3 * np.sign(df['rot_speed']) * ß)/alpha_0
    # ß = np.exp(-abs(df['DXL_Velocity']/q_dot_s)) # assume ∂ = 1
    # df['tau_o'] = (df['U'] - alpha_1 * df['DXL_Velocity'] - alpha_2 * np.sign(df['DXL_Velocity']) * (1 - ß) - alpha_3 * np.sign(df['DXL_Velocity']) * ß)/alpha_0
    



