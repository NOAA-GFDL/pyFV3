from f90nml import Namelist

from ndsl import StencilFactory
from ndsl.constants import K_DIM
from ndsl.stencils.testing import Grid
from pyfv3.stencils import DivergenceDamping
from pyfv3.testing import TranslateDycoreFortranData2Py


class TranslateDivergenceDamping(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid: Grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "u": {},
            "v": {},
            "va": {},
            "damped_rel_vort_bgrid": {"serialname": "vort"},
            "ua": {},
            "divg_d": {},
            "vc": {},
            "uc": {},
            "delpc": {},
            "ke": {},
            "nord_col": {},
            "d2_bg": {},
        }
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = {
            "ke": {"iend": grid.ied + 1, "jend": grid.jed + 1},
            "delpc": {},
        }
        self.max_error = 1.4e-10

    def compute_from_storage(self, inputs):
        nord_col = self.grid.quantity_factory.zeros(dims=[K_DIM], units="unknown")
        nord_col[:] = nord_col.np.asarray(inputs.pop("nord_col"))
        d2_bg = self.grid.quantity_factory.zeros(dims=[K_DIM], units="unknown")
        d2_bg[:] = d2_bg.np.asarray(inputs.pop("d2_bg"))

        # Assumption: see comment in D_SW.init()
        assert self.config.d4_bg_top == self.config.d4_bg_bot

        divdamp = DivergenceDamping(
            self.stencil_factory,
            self.grid.quantity_factory,
            self.grid.grid_data,
            self.grid.damping_coefficients,
            self.grid.nested,
            self.grid.stretched_grid,
            self.config.dddmp,
            self.config.d4_bg_bot,  # assumption d4_bg_top == d4_bg_bot
            self.config.nord,
            self.config.grid_type,
            nord_col,
            d2_bg,
        )
        divdamp(**inputs)
        return inputs
