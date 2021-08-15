from engine import *
from space_mapper_game_objects import *
from engine.standard_game_objects import ButtonPanel, Button
if __name__=='__main__':
    k = KinectCapturer()
    k.transform.translate(0, 0, -3)
    k.transform.rotate(3.14, -0.4, 0)
    #k.transform.translate(-1, 0, 2)
    w = Window(name='Main Window', game_objects=[k])
    w.set_size(1280, 720)
    w.set_position(0, 0)
    

    bp = ButtonPanel(1, 1)
    
    def render_mode_cb(button):
        k.change_render_mode()
        if button.text == 'Show mesh':
            button.update_text('Show point cloud')
        else:
            button.update_text('Show mesh')
    
    render_mode_button = Button(render_mode_cb, 1, 1, 'Show mesh', np.array([0, 1, 0]))
    
    bp.add_button(render_mode_button)
    
    capture_button = Button(None, 1, 1, 'Take capture', np.array([1, 0, 0]))
    commit_button = Button(None, 1, 1, 'Commit', np.array([0, 0, 1]))
    abort_button = Button(None, 1, 1, 'Abort', np.array([1, 0, 0]))
    preview_button = Button(None, 1, 1, 'Show current map', np.array([0, 0, 1]))
    export_button = Button(None, 1, 1, 'Export', np.array([1, 0, 0]))
    
    def commit_cb(button):
        k.commit_capture()
        bp.remove_button(button)
        bp.remove_button(abort_button)
        bp.add_button(preview_button)
        bp.add_button(capture_button)

    def abort_cb(button):
        k.abort_capture()
        bp.remove_button(button)
        bp.remove_button(commit_button)
        bp.add_button(preview_button)
        bp.add_button(capture_button)
        
    def capture_cb(button):
        k.take_capture()
        bp.remove_button(button)
        bp.remove_button(preview_button)
        bp.add_button(commit_button)
        bp.add_button(abort_button)

    def preview_cb(button):
        k.toggle_preview()
        if button.text == 'Show current map':
            button.update_text('Return to capturing')
            bp.remove_button(capture_button)
            bp.add_button(export_button)
        else:
            button.update_text('Show current map')
            bp.remove_button(export_button)
            bp.add_button(capture_button)

    def export_cb(button):
        k.export()

    capture_button.callback = capture_cb
    commit_button.callback = commit_cb
    abort_button.callback = abort_cb
    preview_button.callback = preview_cb
    export_button.callback = export_cb

    bp.add_button(capture_button)
    
    bp.transform.translate(0, 0, -1.3)
    
    
    
    wp = Window(name='Options', game_objects=[bp])
    wp.set_size(1280, 200)
    wp.set_position(0, 730)
    try:
        start_main_loop()
    finally:
        print("lmao")

