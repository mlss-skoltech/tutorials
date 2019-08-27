import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# hardcode landmark indexes
left_brow = np.array([1, 5, 3, 6, 1]) - 1
right_brow = np.array([4, 7, 2, 8, 4]) - 1

left_eye = np.array([9, 13, 11, 14, 9]) - 1
right_eye = np.array([12, 15, 10, 16, 12]) - 1

nosetip = np.array([19, 21, 20, 22, 19]) - 1
mouth_outerlip = np.array([23, 25, 24, 28, 23]) - 1
mouth_innerlip = np.array([23, 26, 24, 27, 23]) - 1

face_outer = np.array([29, 33, 31, 35, 32, 34, 30]) - 1

CONTOURS = [left_brow, right_brow, left_eye, right_eye,
            nosetip, mouth_outerlip, mouth_innerlip, face_outer]


def get_contours(image):
    for contour_idx in CONTOURS:
        x = image[contour_idx, 0]
        y = -image[contour_idx, 1]
        yield x, y



def plot_landmarks(landmarks, ax=None, draw_landmark_id=False, draw_contours=True, draw_landmarks=True,
                   alpha=1, color_landmarks='red', color_contour='orange', get_contour_handles=False):
    """Plots landmarks, connecting them appropriately.
    
    landmarks: ndarray of shape either [35, 2] or [70,]
    ax: axis (created if None)
    """
    if None is ax:
        f = plt.figure(figsize=(8, 8))
        ax = f.gca()

    ax.tick_params(
        axis='both',       # changes apply to both axes
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False, 
        labelleft=False, 
        labelright=False)

    if landmarks.shape == (70,):
        landmarks = landmarks.reshape((35, 2))

    contour_handles = []
    if draw_contours:

        def _plot_landmark(landmarks, idx, color):
            h, = ax.plot(landmarks[idx, 0], -landmarks[idx, 1], color=color)
            return h
        
        contour_handles = [
            _plot_landmark(landmarks, idx, color_contour)
            for idx in CONTOURS
        ]
        
    if draw_landmarks:
        ax.scatter(landmarks[:, 0], -landmarks[:, 1], s=20, color=color_landmarks, alpha=alpha)    
    
    if draw_landmark_id:
        for i in range(35):
            ax.text(s=str(i + 1), x=landmarks[i, 0], y=-landmarks[i, 1])
    
    if get_contour_handles:
        return contour_handles
    

    
def load_data(data_path):
    df = pd.read_csv(data_path + 'kbvt_lfpw_v1_train.csv', delimiter='\t')
    
    # We don't need all of the columns -- only the ones with landmarks
    columns_to_include = [col for col in df.columns.tolist() 
                          if col.endswith('_x') or col.endswith('_y')]
    print('Selecting the following columns from the dataset: {}'.format('\n'.join(columns_to_include)))

    # select only averaged predictions
    data = df[columns_to_include][df['worker'] == 'average']
    landmarks = data.values

    print('\n\n The resulting dataset has shape {}'.format(landmarks.shape))

    return landmarks


def prepare_html_for_scatter_plot(projected_shapes):
    xs = '[' + ','.join(map(str, projected_shapes[:, 0])) + ']'
    ys = '[' + ','.join(map(str, projected_shapes[:, 1])) + ']'
    return f'x: {xs}, y: {ys}'


def prepare_html_for_landmarks(landmarks_for_one_sample):
    landmarks = landmarks_for_one_sample.reshape(35, 2)
    xs = '[' + ',"nan",'.join(','.join(map(str, landmarks[idx, 0])) for idx in CONTOURS) + ']'
    ys = '[' + ',"nan",'.join(','.join(map(str, -landmarks[idx, 1])) for idx in CONTOURS) + ']'
    return f'[{{x: {xs}, y: {ys}, type: "scatter", mode: "lines+markers", line: {{width: 1, color: "orange"}}, marker: {{size: 3, color: "red"}} }}]'


def prepare_html_for_all_landmarks(landmarks):
    return '[' + ','.join(map(prepare_html_for_landmarks, landmarks)) + ']'


def prepare_html_for_visualization(projected_shapes, landmarks, scatterplot_size=[700, 700], annotation_size=[100, 100], floating_annotation=True):
    scatter_data = prepare_html_for_scatter_plot(projected_shapes)
    scatter_width = str(scatterplot_size[0])
    scatter_height = str(scatterplot_size[1])
  
    annotation_data = prepare_html_for_all_landmarks(landmarks)
    annotation_width = str(annotation_size[0])
    annotation_height = str(annotation_size[1])
  
    html = '''
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>.hoverlayer {display: none;}</style>

<div id="figure"></div>
<div id="annotation-figure" style="width: ''' + annotation_width + '''px; height: ''' + annotation_height + '''px; position: absolute; left: 0; top: 0; border: 1px solid #000000; z-index: 999;"></div>


<script>
// find html elements to plot in and hide annotation by default
let figure = document.getElementById('figure'),
    hoverFigure = document.getElementById('annotation-figure');
hoverFigure.style.display = 'none';

// set the data
let projected_shapes = [{
        ''' + scatter_data + ''', type: 'scatter', mode: 'markers', marker: {size: 10, opacity: .5}, hovertemplate: 'amhere'
    }],
    annotation_data = ''' + annotation_data + ''';

// draw the scatter plot
Plotly.plot('figure', projected_shapes, {hovermode: 'closest', width: ''' + scatter_width + ''', height: ''' + scatter_height + '''});

// use this for visualization of annotations
let annotationLayout = {
    xaxis: {
        autorange: true,
        showgrid: false,
        zeroline: false,
        showline: false,
        autotick: true,
        ticks: '',
        showticklabels: false
    },
    yaxis: {
        autorange: true,
        showgrid: false,
        zeroline: false,
        showline: false,
        autotick: true,
        ticks: '',
        showticklabels: false
    },
    margin: {t: 0, l: 0, r: 0, b: 0}
};

let plotAnnotation = function (id) {
    Plotly.plot('annotation-figure', annotation_data[id], annotationLayout);
};

// set the callbacks for annotation visualization''' + (('''
figure.onmousemove = function (event) {
    hoverFigure.style.left = event.pageX + 20 + "px";
    hoverFigure.style.top = event.pageY - 20 - ''' + annotation_height + ''' + "px";
};
''') if floating_annotation else '') + '''

figure.on('plotly_hover', function (data) {
    plotAnnotation(data.points[0].pointIndex);
    hoverFigure.style.display = 'block';
    
})
    .on('plotly_unhover', function (data) {
        hoverFigure.style.display = 'none';
        Plotly.purge('annotation-figure');
    });
</script>'''
    return html



    
__all__ = [
    "load_data",
    "plot_landmarks",
    "prepare_html_for_visualization",
]