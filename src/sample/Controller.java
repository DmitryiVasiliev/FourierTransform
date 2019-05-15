package sample;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.opencv.imgproc.Imgproc;

public class Controller {
    // images to show in the view
    @FXML
    private ImageView originalImage;
    @FXML
    private ImageView transformedImage;
    @FXML
    private ImageView antitransformedImage;
    // a FXML button for performing the transformation
    @FXML
    private Button transformButton;
    // a FXML button for performing the antitransformation
    @FXML
    private Button antitransformButton;

    // the main stage
    private Stage stage;
    // the JavaFX file chooser
    private FileChooser fileChooser;
    // support variables
    private Mat image;
    private List<Mat> planes;
    // the final complex image
    private Mat complexImage;


    /**
     * Init the needed variables
     */
    protected void init() {
        this.fileChooser = new FileChooser();
        this.image = new Mat();
        this.planes = new ArrayList<>();
        this.complexImage = new Mat();
    }

    /**
     * Load an image from disk
     */
    @FXML
    protected void loadImage() {
        // show the open dialog window
        File file = this.fileChooser.showOpenDialog(this.stage);
        if (file != null) {
            // read the image in gray scale
            this.image = Imgcodecs.imread(file.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
            //create Hole 20х20
            Point p1 = new Point(0, 0);
            Point p2 = new Point(19, 19);
            this.updateImageView(originalImage, Utils.mat2Image(this.image));
            Imgproc.rectangle(this.image, p1, p2, new Scalar(0, 0, 0), -1);

            // show the image


            // set a fixed width
            this.originalImage.setFitWidth(250);
            // preserve image ratio
            this.originalImage.setPreserveRatio(true);
            // update the UI
            this.transformButton.setDisable(false);
            for (int i = 0; i < 120; i++) {
                transformImage();
                changeSpectrumComponents(i);
                antitransformImage();
                Imgcodecs.imwrite(i + "test.png", this.image);
            }
            // empty the image planes and the image views if it is not the first
            // loaded image
            if (!this.planes.isEmpty()) {
                this.planes.clear();
                this.transformedImage.setImage(null);
                this.antitransformedImage.setImage(null);
            }

        }
    }

    protected void changeSpectrumComponents(int i) {
        int count = 0;
        int size = (int) (this.image.size().height / 2 + 1);
        ArrayList<double[]> doubles = new ArrayList<>();
        for (int j = size - 1 - i; j < size + i; j++)
            for (int k = size - 1 - i; k < size + i; k++) {
                doubles.add(complexImage.get(j, k));
            }
        for (int j = 0; j < this.image.size().height; j++)
            for (int k = 0; k < this.image.size().height; k++) {
                complexImage.put(j, k, 0, 0);
            }
        for (int j = size - 1 - i; j < size + i; j++)
            for (int k = size - 1 - i; k < size + i; k++) {
                complexImage.put(j, k, doubles.get(count));
                count++;
            }
    }

    /**
     * The action triggered by pushing the button for apply the dft to the
     * loaded image
     */
    int c = 0;

    @FXML
    protected void transformImage() {
        this.planes.clear();
        // optimize the dimension of the loaded image
        Mat padded = this.optimizeImageDim(this.image);
        padded.convertTo(padded, CvType.CV_32F);
        // prepare the image planes to obtain the complex image
        this.planes.add(padded);
        this.planes.add(Mat.zeros(padded.size(), CvType.CV_32F));
        // prepare a complex image for performing the dft
        Core.merge(this.planes, this.complexImage);

        // dft
        Core.dft(this.complexImage, this.complexImage);

        // optimize the image resulting from the dft operation
        Mat magnitude = this.createOptimizedMagnitude(this.complexImage);

        // show the result of the transformation as an image
            this.updateImageView(transformedImage, Utils.mat2Image(magnitude));
        // set a fixed width
        this.transformedImage.setFitWidth(250);
        // preserve image ratio
        this.transformedImage.setPreserveRatio(true);

        // enable the button for performing the antitransformation
        this.antitransformButton.setDisable(false);
        // disable the button for applying the dft
        this.transformButton.setDisable(true);
    }



    @FXML
    protected void antitransformImage() {
//        Core.copyTo(complexImage, complexImage, mask);
        Core.idft(this.complexImage, this.complexImage);

        Mat restoredImage = new Mat();
        Core.split(this.complexImage, this.planes);
        Core.normalize(this.planes.get(0), restoredImage, 0, 255, Core.NORM_MINMAX);

        // move back the Mat to 8 bit, in order to proper show the result
        restoredImage.convertTo(restoredImage, CvType.CV_8U);

        for (int j = 0; j < 20; j++)
            for (int k = 0; k < 20; k++) {
                this.image.put(j, k, restoredImage.get(j, k));
            }

        //  this.updateImageView(antitransformedImage, Utils.mat2Image(restoredImage));
        // set a fixed width
        this.antitransformedImage.setFitWidth(250);
        // preserve image ratio
        this.antitransformedImage.setPreserveRatio(true);

        // disable the button for performing the antitransformation
        this.antitransformButton.setDisable(true);
    }

    private Mat optimizeImageDim(Mat image) {
        // init
        Mat padded = new Mat();
        // get the optimal rows size for dft
        int addPixelRows = Core.getOptimalDFTSize(image.rows());
        // get the optimal cols size for dft
        int addPixelCols = Core.getOptimalDFTSize(image.cols());
        // apply the optimal cols and rows size to the image
        Core.copyMakeBorder(image, padded, 0, addPixelRows - image.rows(), 0, addPixelCols - image.cols(),
                Core.BORDER_CONSTANT, Scalar.all(0));

        return padded;
    }

    private Mat createOptimizedMagnitude(Mat complexImage) {
        // init
        List<Mat> newPlanes = new ArrayList<>();
        Mat mag = new Mat();
        // split the comples image in two planes
        Core.split(complexImage, newPlanes);
        // compute the magnitude
        Core.magnitude(newPlanes.get(0), newPlanes.get(1), mag);

        // move to a logarithmic scale
        Core.add(Mat.ones(mag.size(), CvType.CV_32F), mag, mag);
        Core.log(mag, mag);
        // optionally reorder the 4 quadrants of the magnitude image
        this.shiftDFT(mag);
        // normalize the magnitude image for the visualization since both JavaFX
        // and OpenCV need images with value between 0 and 255
        // convert back to CV_8UC1
        mag.convertTo(mag, CvType.CV_8UC1);
        Core.normalize(mag, mag, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);

        // you can also write on disk the resulting image...
        // Imgcodecs.imwrite("../magnitude.png", mag);

        return mag;
    }

    private void shiftDFT(Mat image) {
        image = image.submat(new Rect(0, 0, image.cols() & -2, image.rows() & -2));
        int cx = image.cols() / 2;
        int cy = image.rows() / 2;

        Mat q0 = new Mat(image, new Rect(0, 0, cx, cy));
        Mat q1 = new Mat(image, new Rect(cx, 0, cx, cy));
        Mat q2 = new Mat(image, new Rect(0, cy, cx, cy));
        Mat q3 = new Mat(image, new Rect(cx, cy, cx, cy));

        Mat tmp = new Mat();
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);
    }


    public void setStage(Stage stage) {
        this.stage = stage;
    }



    private void updateImageView(ImageView view, Image image) {
        Utils.onFXThread(view.imageProperty(), image);
    }

}
