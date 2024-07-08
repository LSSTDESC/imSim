from galsim.config import StampBuilder, RegisterStampType
from galsim.config import WCSBuilder, RegisterWCSType
from galsim.config.input import GetInputObj


class CBPWCS(WCSBuilder):
    def buildWCS(self, config, base, logger):
        rubin = GetInputObj('telescope', config, base, 'telescope', 0).fiducial
        cbp = GetInputObj('telescope', config, base, 'telescope', 1).fiducial

        import ipyvolume as ipv
        import webbrowser
        import tempfile
        fig = ipv.figure(width=1200, height=800)
        ipv.style.set_style_dark()
        ipv.style.box_off()
        rubin.draw3d(ipv, color='cyan')
        cbp.draw3d(ipv, color='magenta')
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
            html_file = tmpfile.name
            # Save the figure to the temporary HTML file
            ipv.save(html_file)
        webbrowser.open(f'file://{html_file}')



class CBPStampBuilder(StampBuilder):
    pass


RegisterStampType('cbp', CBPStampBuilder())
RegisterWCSType('cbp', CBPWCS(), input_type="telescope")
