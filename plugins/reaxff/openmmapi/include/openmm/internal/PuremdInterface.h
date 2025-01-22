#ifndef OPENMM_PUREMDINTERFACE_H
#define OPENMM_PUREMDINTERFACE_H
#define QMMM

#include <string>
#include <vector>


/**
 * An interface class to the C API of ReaxFF-PuReMD.
 */
class PuremdInterface
{
  public:
    PuremdInterface();
    ~PuremdInterface();
    
    /**
     * Set the input files for ReaxFF-PuReMD
     *
     * @param ffield_filename The force field parameter file path
     * @param control_filename The control file path
     */
    void setInputFileNames(const std::string &ffield_filename,
                           const std::string &control_filename);
    /**
     * Get the forces from ReaxFF-PuReMD.
     *
     * @param num_qm_atoms The number of QM atoms
     * @param qm_symbols The list of QM atom symbols, has size 2*num_qm_atoms
     * @param qm_pos The positions of the QM atoms
     * @param num_mm_atoms The number of MM atoms
     * @param mm_symbols The MM symbols, has size 2*num_mm_atoms
     * @param mm_pos_q The positions and charges of the MM atoms
     * @param sim_box_info The simulation box sizes and the angles
     * @param qm_forces The output vector of the forces on the QM atoms
     * @param mm_forces The output array of the forces on the MM atoms
     * @param qm_q The calculated charges of the QM atoms
     * @param totalEnergy The energy of the QM region
     */
    void getReaxffPuremdForces(int                        num_qm_atoms,
                               const std::vector<char>   &qm_symbols,
                               const std::vector<double> &qm_pos,
                               int                        num_mm_atoms,
                               const std::vector<char>   &mm_symbols,
                               const std::vector<double> &mm_pos_q,
                               const std::vector<double> &sim_box_info,
                               std::vector<double>       &qm_forces,
                               std::vector<double>       &mm_forces,
                               std::vector<double> &qm_q, double &totalEnergy);

  private:
    bool                      firstCall;
    void                     *handlePuremd;
    int                       retPuremd;
    const std::vector<double> sim_box_info;
    std::string               ffield_filename;
    std::string               control_filename;
};

#endif // OPENMM_PUREMDINTERFACE_H
