import numpy as np
import pandas as pd
import cmath
import os
import sys

# Configuración del Sistema Base (Página 4 del PDF)
MVA_BASE = 100.0
KV_BASE = 230.0

# Cálculo de Bases
I_BASE = (MVA_BASE * 1e6) / (np.sqrt(3) * KV_BASE * 1e3) # Amperes
Z_BASE = (KV_BASE**2) / MVA_BASE # Ohms

class PowerSystem:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.y_bus = np.zeros((num_nodes, num_nodes), dtype=complex)
        self.z_bus = None
        self.elements = []

    def add_element(self, n_from, n_to, x_pu, is_generator=False):
        """
        Agrega una línea o generador.
        n_from, n_to: Números de nodo (1-based). Usar 0 para referencia (tierra).
        x_pu: Reactancia en por unidad.
        """
        z = complex(0, x_pu)
        y = 1 / z
        self.elements.append({
            'from': n_from, 'to': n_to, 'x': x_pu, 'y': y, 'type': 'GEN' if is_generator else 'LINE'
        })

        # Ajuste de índices para matriz (0-based)
        i = n_from - 1
        j = n_to - 1

        if is_generator:
            # Generador conecta el nodo a tierra (referencia)
            # Solo afecta la diagonal del nodo
            if n_from > 0:
                self.y_bus[i, i] += y
        else:
            # Línea de transmisión entre dos nodos
            if n_from > 0 and n_to > 0:
                self.y_bus[i, i] += y
                self.y_bus[j, j] += y
                self.y_bus[i, j] -= y
                self.y_bus[j, i] -= y
            # Si fuera una rama shunt a tierra (no aplica en este ejemplo específico pero el código lo soportaría)
            elif n_from > 0: 
                self.y_bus[i, i] += y

    def build_zbus(self):
        print(f"\n[INFO] Invirtiendo Ybus para obtener Zbus...")
        try:
            self.z_bus = np.linalg.inv(self.y_bus)
            return True
        except np.linalg.LinAlgError:
            print("[ERROR] La matriz Ybus es singular y no se puede invertir (sistema aislado).")
            return False

    def calculate_short_circuit(self, fault_node_idx):
        """
        Calcula falla trifásica en el nodo especificado.
        """
        k = fault_node_idx - 1 # Ajuste a índice 0
        
        # 1. Corriente de Falla (Icc)
        # I_f (pu) = V_f_pre (1.0) / Z_kk
        z_kk = self.z_bus[k, k]
        v_pre = 1.0 + 0j
        i_cc_pu = v_pre / z_kk
        i_cc_amp = abs(i_cc_pu) * I_BASE

        print(f"\n{'='*60}")
        print(f" RESULTADOS DE FALLA EN NODO {fault_node_idx}")
        print(f"{'='*60}")
        print(f"Zqq (Thévenin en nodo falla): {z_kk:.4f} pu")
        print(f"Icc (pu) Magnitude:           {abs(i_cc_pu):.4f} pu")
        print(f"Icc (Amp) Magnitude:          {i_cc_amp:.2f} A")
        print(f"-"*60)

        # 2. Voltajes Post-Falla en todos los nodos
        # V_i = V_pre - Z_ik * I_cc
        v_post = np.zeros(self.num_nodes, dtype=complex)
        print("\n>>> VOLTAJES POST-FALLA (Nodos):")
        print(f"{'Nodo':<10} | {'Voltaje (pu)':<15} | {'Ángulo (°)'}")
        print("-" * 45)
        
        for i in range(self.num_nodes):
            z_ik = self.z_bus[i, k]
            v_post[i] = v_pre - (z_ik * i_cc_pu)
            ang = np.degrees(cmath.phase(v_post[i]))
            print(f"{i+1:<10} | {abs(v_post[i]):.4f} pu      | {ang:.2f}°")

        # 3. Corrientes de Rama (Contribuciones)
        # I_ij = (V_i - V_j) / z_linea
        print("\n>>> CORRIENTES DE RAMA (Contribuciones):")
        print(f"{'Desde':<6} {'Hacia':<6} | {'I rama (pu)':<15} | {'I rama (Amp)'}")
        print("-" * 55)
        
        for el in self.elements:
            n_i = el['from']
            n_j = el['to']
            y_line = el['y']
            
            # Voltaje en i
            v_i = v_post[n_i-1] if n_i > 0 else 0
            # Voltaje en j
            v_j = v_post[n_j-1] if n_j > 0 else 0
            
            # Corriente
            i_branch = (v_i - v_j) * y_line
            i_branch_amp = abs(i_branch) * I_BASE
            
            # Solo mostrar ramas conectadas (ignorar generadores si se desea solo ver líneas, 
            # pero el profe pide "cada rama", así que incluimos todo)
            print(f"N{n_i:<5} N{n_j:<5} | {abs(i_branch):.4f} pu      | {i_branch_amp:.2f} A")

    def print_matrices(self):
        print("\n>>> MATRIZ Y-BUS (Admitancias):")
        # Formato limpio usando DataFrame de pandas
        df_y = pd.DataFrame(self.y_bus, 
                            index=[f"N{i+1}" for i in range(self.num_nodes)],
                            columns=[f"N{i+1}" for i in range(self.num_nodes)])
        print(df_y.map(lambda x: f"{x.imag:+.4f}j"))

        print("\n>>> MATRIZ Z-BUS (Impedancias):")
        df_z = pd.DataFrame(self.z_bus, 
                            index=[f"N{i+1}" for i in range(self.num_nodes)],
                            columns=[f"N{i+1}" for i in range(self.num_nodes)])
        print(df_z.map(lambda x: f"{x.imag:+.4f}j"))

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=== ESIME Z-BUS SOLVER ===")
    print("Materia: Computación Aplicada a Sist. Eléctricos")
    print(f"Base: {MVA_BASE} MVA, {KV_BASE} kV -> Ibase: {I_BASE:.2f} A\n")

    # Inicializar sistema de 3 nodos (según PDF)
    ps = PowerSystem(3)

    # DATOS DEL PDF (Página 4)
    # Generadores (conectados a tierra/referencia virtual 0)
    # G1 en N1, X=0.2
    ps.add_element(1, 0, 0.2, is_generator=True)
    # G2 en N2, X=0.1428
    ps.add_element(2, 0, 0.1428, is_generator=True)

    # Líneas de Transmisión
    # L1: N1 - N2, X=0.5
    ps.add_element(1, 2, 0.5)
    # L2: N1 - N3, X=0.3333
    ps.add_element(1, 3, 0.3333)
    # L3: N2 - N3, X=0.25
    ps.add_element(2, 3, 0.25)

    # Cálculos iniciales
    ps.build_zbus()
    ps.print_matrices()

    while True:
        try:
            print("\n" + "="*30)
            sel = input("Ingrese el Nodo de Falla (1-3) o 'q' para salir: ")
            if sel.lower() == 'q':
                break
            node = int(sel)
            if 1 <= node <= 3:
                ps.calculate_short_circuit(node)
                input("\nPresione Enter para continuar...")
                os.system('cls' if os.name == 'nt' else 'clear')
                ps.print_matrices() # Volver a mostrar matrices para referencia
            else:
                print("Nodo fuera de rango (1-3).")
        except ValueError:
            print("Entrada inválida.")

if __name__ == "__main__":
    main()
