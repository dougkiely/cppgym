#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <iostream> // debug
#include <memory>
#include <numbers>
#include <vector>

#include <GL/gl.h>
#include <GL/freeglut.h>

namespace gym::envs::classic_control {

// TODO: check for GL errors by calling glGetError() after each gl*() call
// TODO: check for GLUT errors after each glut*() call
// TODO: check for const-correctness in this code, although it's likely only a few members functions code in here is const (it's all mutable data and function calls)
// TODO: add any appropriate noexcept clauses

/*display_type get_display(spec_type& spec) {
    return display_type {spec};
}

window_type get_window(uint64_t width, uint64_t height, display_type display, char** args) {
    return window_type {width, height, display, args};
}*/

template<typename T>
concept attr_concept = requires(T t) {
    t.enable(); // TODO: should enable() be changed to show()? Probably not.
    t.disable();// TODO: should disable() be changed to hide()? Probably not.
    { t.to_string() } -> std::convertible_to<std::string>;
};

template<typename T>
concept point_2d_concept = std::default_initializable<T> && std::constructible_from<T, typename T::value_type, typename T::value_type> && requires(T t) {
    { t.x } -> std::convertible_to<typename T::value_type>;
    { t.y } -> std::convertible_to<typename T::value_type>;
};

template<typename T>
concept point_3d_concept = std::default_initializable<T> && std::constructible_from<T, typename T::value_type, typename T::value_type, typename T::value_type> && requires(T t) {
    { t.x } -> std::convertible_to<typename T::value_type>;
    { t.y } -> std::convertible_to<typename T::value_type>;
    { t.z } -> std::convertible_to<typename T::value_type>;
};

template<typename T>
class point_2d {
    public:
        using value_type = T;
        T x;
        T y;
        point_2d()=default;
        point_2d(T x_, T y_) : x(x_), y(y_) {}
        point_2d(const point_2d& other)=default;
        point_2d& operator=(const point_2d& other)=default;
        point_2d(point_2d& other)=default;
        point_2d& operator=(point_2d& other)=default;
};

template<typename T>
class point_3d {
    public:
        using value_type = T;
        T x;
        T y;
        T z;
        point_3d()=default;
        point_3d(T x_, T y_, T z_) : x(std::move(x_)), y(std::move(y_)), z(std::move(z_)) {}
        point_3d(const point_3d& other)=default;
        point_3d& operator=(const point_3d& other)=default;
        point_3d(point_3d& other)=default;
        point_3d& operator=(point_3d& other)=default;
};

/*template<std::floating_point T>
concept scale_concept = requires {
    public:
        scale_type(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
        T x{};
        T y{};
        T z{};
};*/

class Attr {
    public:
        virtual void enable()=0;
        virtual void disable() {}
        virtual std::string to_string() const=0;
        virtual ~Attr()=default;
};

class Color : public Attr {
    public:
        using value_type = GLfloat;

        Color(value_type red, value_type green, value_type blue)
            : red_(red), green_(green), blue_(blue), alpha_(1.0f) {}

        Color(value_type red, value_type green, value_type blue, value_type alpha)
            : red_(red), green_(green), blue_(blue), alpha_(alpha) {}

        auto get_color() -> std::tuple<value_type, value_type, value_type> {
            return {red_, green_, blue_};
        }

        void set_color(value_type red, value_type green, value_type blue) {
            red_ = red; green_ = green; blue_ = blue;
        }

        void set_color(value_type red, value_type green, value_type blue, value_type alpha) {
            red_ = red; green_ = green; blue_ = blue; alpha_ = alpha;
        }

        void enable() override {
            glColor4f(red_, green_, blue_, alpha_);
        }

        std::string to_string() const override {
            return "Color: red=" + std::to_string(red_) + ", green=" + std::to_string(green_) + ", blue=" + std::to_string(blue_) + ", alpha=" + std::to_string(alpha_);
        }

    private:
        value_type red_{};
        value_type green_{};
        value_type blue_{};
        value_type alpha_{};
};

class Transform : public Attr {
    public:
        using translation_type = point_2d<GLfloat>;
        using rotation_type = GLfloat;
        using scale_type = point_2d<GLfloat>;

        Transform()=default;
        Transform(translation_type translation) : translation_(translation) {}
        Transform(translation_type translation, std::floating_point auto rotation, scale_type scale) : translation_(translation), rotation_(rotation), scale_(scale) {}

        void enable() override {
            glPushMatrix();
            glTranslatef(translation_.x, translation_.y, 0.0f);
            glRotatef(RAD2DEG * rotation_, 0.0f, 0.0f, 1.0f);
            glScalef(scale_.x, scale_.y, 1.0f);
        }

        void disable() override {
            glPopMatrix();
        }

        void set_translation(const point_2d_concept auto& pt_2d) {
            //std::cout << "calling Transform::set_translation...\n"; // TODO: add as debug info?
            //translation_ = std::move(pt_2d); // crash occurred, don't std::move() from here
            translation_.x = pt_2d.x;
            translation_.y = pt_2d.y;
            //std::cout << "done calling Transform::set_translation\n"; // TODO: add as debug info?
        }

        void set_rotation(const std::floating_point auto rot) {
            //std::cout << "calling Transform::set_rotation...\n"; // TODO: add as debug info?
            rotation_ = rot;
            //std::cout << "done calling Transform::set_rotation\n"; // TODO: add as debug info?
        }

        void set_scale(const point_2d_concept auto& pt_2d) {
            //std::cout << "calling Transform::set_scale...\n"; // TODO: add as debug info?
            //scale_ = std::move(pt_2d);
            scale_.x = pt_2d.x;
            scale_.y = pt_2d.y;
            //std::cout << "done calling Transform::set_scale\n"; // TODO: add as debug info?
        }

        std::string to_string() const override {
            return "Transform: translation.x=" + std::to_string(translation_.x) + ", translation.y=" + std::to_string(translation_.y) + ", rotation=" + std::to_string(rotation_) + ", scale.x=" + std::to_string(scale_.x) + ", scale.y=" + std::to_string(scale_.y);
        }

    private:
        const rotation_type RAD2DEG{57.29577951308232f};

        translation_type translation_{0.0f, 0.0f};
        rotation_type rotation_{0.0f};
        scale_type scale_{1.0f, 1.0f};
};

class LineStyle : public Attr {
    public:
        using factor_type = GLint;
        using pattern_type = GLushort;

        LineStyle(pattern_type pattern) : LineStyle(1.0f, pattern) {}
        LineStyle(factor_type factor, pattern_type pattern) : factor_(factor), pattern_(pattern) {}

        void enable() override {
            glEnable(GL_LINE_STIPPLE);
            glLineStipple(factor_, pattern_);
        }

        void disable() {
            glDisable(GL_LINE_STIPPLE);
        }

        std::string to_string() const override {
            return "LineStyle: factor=" + std::to_string(factor_) + ", pattern=" + std::to_string(pattern_);
        }

    private:
        factor_type factor_{};
        pattern_type pattern_{};
};

class LineWidth : public Attr {
    public:
        using value_type = GLfloat;

        LineWidth(value_type width) : width_(width) {}

        void set_width(value_type width) {
            width_ = width;
        }

        void enable() override {
            glLineWidth(width_);
        }

        std::string to_string() const override {
            return "LineWidth: width=" + std::to_string(width_);
        }

    private:
        value_type width_{};
};

template<typename T>
concept geom_concept = requires(T t, Color::value_type color) {
    t.render();
    //t.add_attr(attr_concept auto... attrs);
    t.set_color(color, color, color);
};

class Geom {
    public:
        Geom(bool add_default_attrs=true) {
            if (add_default_attrs)
                attrs_.push_back(&color_);
        }

        void render() {
            /*std::for_each(std::rbegin(attrs_), std::rend(attrs_), [](Attr& attr){
                for (auto& attr : attrs_)
                    attr.get().enable();
            });*/
            std::for_each(std::rbegin(attrs_), std::rend(attrs_), [](Attr* attr){
                attr->enable();
            });

            render1();

            /*std::for_each(std::begin(attrs_), std::end(attrs_), [](Attr& attr){
                for (auto& attr : attrs_)
                    attr.get().disable();
            });*/
            std::for_each(std::begin(attrs_), std::end(attrs_), [](Attr* attr){
                attr->disable();
            });
        }

        /*template<attr_concept Attribute, typename... Args>
        void add_attr(Args&&... args) {
            attrs_.push_back(std::move(std::make_unique<Attribute>(std::forward<Args>(args)...)));
        }*/

        //void add_attr(attr_concept auto& attr) {
        /*void add_attr(Attr& attr) {
            attrs_.push_back(std::ref(attr));
        }*/

        void add_attr(Attr& attr) {
            attrs_.push_back(&attr);
        }

        /*void add_attr(Attr&& attr) {
            attrs_.push_back();
        }*/

        /*template<typename T>
        void remove_attr() {
            std::erase_if(attrs_, [](auto& x) { return typeid(*x) == typeid(T); });
        }*/

        void set_color(Color::value_type red, Color::value_type green, Color::value_type blue) {
            color_.set_color(red, green, blue);
        }

        virtual std::string to_string() const {
            std::string s{"Geom:\n"};

            for (const auto& attr : attrs_)
                s += attr->to_string() + '\n';

            return s;
        }

    protected:
        virtual void render1()=0;

    private:
        Color color_{0.0f, 0.0f, 0.0f, 1.0f};
        //std::vector<std::reference_wrapper<Attr>> attrs_;
        std::vector<Attr*> attrs_;
};

class Compound : public Geom {
    public:
        using value_type = std::vector<Geom>;

        //Compound(std::initializer_list<Geom> geoms) : Compound(std::move(static_cast<std::vector<Geom>>(geoms))) {}
        //Compound(std::initializer_list<Geom> geoms) {
        //    geoms_.insert(std::end(geoms_), std::begin(geoms), std::end(geoms));
        //}

        Compound(std::vector<Geom> geoms) : geoms_(std::move(geoms)) {}

    protected:
        void render1() override {
            for (auto& g : geoms_)
                g.render();
        }

    private:
        std::vector<Geom> geoms_;
};

class Image : public Geom {
    public:
        using value_type = GLfloat;

        Image(std::string_view fname, value_type width, value_type height) : width_(width), height_(height) {
            set_color(1.0, 0.5, 0.5); //1.0, 1.0, 1.0
            std::cout << "loading image: " << fname << '\n';
            //img_ = image.load(fname);
        }

    protected:
        void render1() override {
            //img_.blit(-width/2, -height/2, width, height); //glBlitFrameBuffer
        }

    private:
        const value_type width_; //uint64_t width_{};
        const value_type height_; //uint64_t height_{};
        //??? img_;
        //bool flip_{false};
};

class PolyLine : public Geom {
    public:
        using value_type = point_2d<GLfloat>;

        PolyLine()=default;
        PolyLine(const bool close) : close_(close) {}
        //PolyLine& operator=(const PolyLine&)=default;
        //PolyLine& operator=(PolyLine&&)=default;

        PolyLine(std::initializer_list<value_type> vertices) : PolyLine(std::move(vertices), false) {}

        PolyLine(std::initializer_list<value_type> vertices, const bool close, const LineWidth::value_type width=1) : close_(close) {
            vertices_.insert(std::end(vertices_), std::begin(vertices), std::end(vertices));
            line_width_.set_width(width);
            add_attr(line_width_);
        }

        PolyLine(std::vector<value_type> vertices) : PolyLine(std::move(vertices), false) {}

        PolyLine(std::vector<value_type> vertices, const bool close, const LineWidth::value_type width=1) : vertices_(std::move(vertices)), close_(close) {
            line_width_.set_width(width);
            add_attr(line_width_);
        }

        void set_vertices(std::initializer_list<value_type> vertices) {
            vertices_.clear();
            vertices_.insert(std::end(vertices_), std::begin(vertices), std::end(vertices));
        }

        void set_vertices(std::vector<value_type> vertices) {
            vertices_ = std::move(vertices);
        }

        void set_width(LineWidth::value_type width) {
            line_width_.set_width(width);
        }

        std::string to_string() const override {
            std::string s("PolyLine: {\n");
            for (const auto& vertex : vertices_)
                s += "vertex.x=" + std::to_string(vertex.x) + ", vertex.y=" + std::to_string(vertex.y) + "\n";
            s += "close=" + std::string(close_ ? "true" : "false") + "\n";
            s += "line_width=" + line_width_.to_string() + "\n";
            s += "}";
            return s;
        }

    protected:
        void render1() override {
            glBegin(close_ ? GL_LINE_LOOP : GL_LINE_STRIP);
            // note: don't call glGetError() between glBegin() and glEnd()
            for (const auto& vertex : vertices_) {
                glVertex3f(vertex.x, vertex.y, 0.0f);
                // note: don't call glGetError() between glBegin() and glEnd()
            }
            glEnd();

            GLenum error{GL_NO_ERROR};
            do {
                error = glGetError();
                if (error != GL_NO_ERROR)
                    std::cout << "PolyLine::render1: glEnd: error=" << error << '\n';
            } while (error != GL_NO_ERROR);
        }

    private:
        std::vector<value_type> vertices_;
        bool close_{false};
        LineWidth line_width_{1};
};

class Line : public Geom {
    public:
        using value_type = point_2d<GLfloat>;

        Line() : Line({0.0f, 0.0f}, {0.0f, 0.0f}) {}

        Line(value_type start, value_type end) : start_(start), end_(end) {
            add_attr(line_width_);
        }

        void set_width(LineWidth::value_type width) {
            line_width_.set_width(width);
        }

        void set_points(value_type start, value_type end) {
            start_ = std::move(start);
            end_ = std::move(end);
        }

        std::string to_string() const override {
            return "Line: start.x=" + std::to_string(start_.x) + ", start.y=" + std::to_string(start_.y) + ", end.x=" + std::to_string(end_.x) + ", end.y=" + std::to_string(end_.y) + "\n inherits from: {\n" + Geom::to_string() + '}';
        }

    protected:
        void render1() override {
            glBegin(GL_LINES);
            glVertex2f(start_.x, start_.y);
            glVertex2f(end_.x, end_.y);
            glEnd();
        }

    private:
        value_type start_{0.0f, 0.0f};
        value_type end_{0.0f, 0.0f};
        LineWidth line_width_{1};
};

class Point : public Geom {
    protected:
        void render1() override {
            glBegin(GL_POINTS);
            glVertex3f(0.0f, 0.0f, 0.0f);
            glEnd();
        }
};

class FilledPolygon : public Geom {
    public:
        using value_type = point_2d<GLfloat>;

        FilledPolygon()=default;

        FilledPolygon(std::initializer_list<value_type> vertices) {
            vertices_.insert(std::end(vertices_), std::begin(vertices), std::end(vertices));
        }

        FilledPolygon(std::vector<value_type> vertices) : vertices_(std::move(vertices)) {}

        void set_vertices(std::initializer_list<value_type> vertices) {
            vertices_.clear();
            vertices_.insert(std::end(vertices_), std::begin(vertices), std::end(vertices));
        }

        void set_vertices(std::vector<value_type> vertices) {
            vertices_ = std::move(vertices);
        }

        std::string to_string() const override {
            std::string s("FilledPolygon: {\n");
            for (const auto& vertex : vertices_)
                s += "vertex.x=" + std::to_string(vertex.x) + ", vertex.y=" + std::to_string(vertex.y) + "\n";
            s += "}";
            return s;
        }

    protected:
        void render1() override {
            GLenum mode{GL_POLYGON};

            if (vertices_.size() == 4)
                mode = GL_QUADS;
            else if (vertices_.size() < 4)
                mode = GL_TRIANGLES;

            glBegin(mode);
            for (const auto& vertex : vertices_)
                glVertex3f(vertex.x, vertex.y, 0.0f);
            glEnd();
        }

    private:
        std::vector<value_type> vertices_;
};

//template<std::floating_point T=GLfloat>
//typedef vertices_2d = std::vector<point_2d<T>>;
typedef std::vector<point_2d<GLfloat>> vertices_2d;

PolyLine make_polyline(vertices_2d vertices) {
    return PolyLine(vertices, false);
}

template<typename T=FilledPolygon>
T make_polygon(vertices_2d vertices)
    requires requires {
        std::same_as<T, FilledPolygon> || std::same_as<T, PolyLine>;
    }
{
    if constexpr (std::same_as<T, FilledPolygon>)
        return FilledPolygon(vertices);
    else
        return PolyLine(vertices, true);
}

//template<geom_concept auto T>
template<typename T=FilledPolygon>
T make_circle(const GLfloat radius=10.0f, const uint64_t resolution=30)
    requires requires {
        std::same_as<T, FilledPolygon> || std::same_as<T, PolyLine>; } {
//std::unique_ptr<Geom> make_circle(const GLfloat radius=10.0f, const uint64_t resolution=30, const bool filled=true) {
//std::unique_ptr<Geom> make_circle(std::floating_point auto const radius=10.0f, std::unsigned_integral auto const resolution=30, const bool filled=true) {

    vertices_2d vertices;

    for (uint64_t i=0; i < resolution; ++i) {
        auto angle = 2 * std::numbers::pi_v<GLfloat> * static_cast<GLfloat>(i) / static_cast<GLfloat>(resolution);
        vertices.emplace_back(std::cos(angle) * radius, std::sin(angle) * radius);
    }

    return make_polygon<T>(std::move(vertices));
}

class Capsule : public Geom {
    public:
        using value_type = GLfloat;

        Capsule(const value_type length, const value_type width) : length_(length), width_(width) {
            circ1_.add_attr(tf_);
        }

        void set_color(Color::value_type red, Color::value_type green, Color::value_type blue) {
            box_.set_color(red, green, blue);
            circ0_.set_color(red, green, blue);
            circ1_.set_color(red, green, blue);
        }

    protected:
        void render1() override {
            box_.render();
            circ0_.render();
            circ1_.render();
        }

    private:
        const value_type length_;
        const value_type width_;
        const value_type l_{0.0};
        const value_type r_{length_};
        const value_type t_{width_/2};
        const value_type b_{-width_/2};

        FilledPolygon box_{{l_, b_}, {l_, t_}, {r_, t_}, {r_, b_}};
        FilledPolygon circ0_{make_circle(t_)};
        FilledPolygon circ1_{make_circle(t_)};
        Transform tf_{{length_, 0.0}};
};

/*Compound make_capsule(const GLfloat length, const GLfloat width) {
    const GLfloat l{0.0}, r{length}, t{width/2}, b{-width/2};
    auto box{make_polygon(std::move(vertices_2d{{l, b}, {l, t}, {r, t}, {r, b}}))};
    auto circ0{make_circle(width/2)};
    auto circ1{make_circle(width/2)};
    Transform tf{{length, 0.0}};
    circ1.add_attr(tf);
    return Compound(std::move(std::vector<Geom>{std::move(box), std::move(circ0), std::move(circ1)}));
}*/

Capsule make_capsule(const GLfloat length, const GLfloat width) {
    return Capsule(length, width);
}

static void init_display(int argc, char** argv, const std::unsigned_integral auto width, const std::unsigned_integral auto height, const std::integral auto x, const std::integral auto y);

static void init_display(int argc, char** argv, const std::unsigned_integral auto width, const std::unsigned_integral auto height) {
    init_display(argc, argv, width, height, -1, -1);
}

class Display {
    friend void init_display(int, char**, const std::unsigned_integral auto, const std::unsigned_integral auto, const std::integral auto, const std::integral auto);
    protected:
        Display(int argc, char** argv, const std::unsigned_integral auto width, const std::unsigned_integral auto height, const std::integral auto x, const std::integral auto y) {
            glutInit(&argc, argv);
            glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE | GLUT_DEPTH);
            glutInitWindowSize(width, height);
            glutInitWindowPosition(x, y);
        }
};

static void init_display(int argc, char** argv, const std::unsigned_integral auto width, const std::unsigned_integral auto height, const std::integral auto x, const std::integral auto y) {
    static Display display(argc, argv, width, height, x, y);
}

/*extern "C" {

static void close_func(void) {
    std::cout << "WINDOW CLOSED CALLBACK!\n";
}

}*/

class Window {
    public:
        Window(std::string_view title, const std::unsigned_integral auto width, const std::unsigned_integral auto height) : Window(title, width, height, -1, -1) {}

        Window(std::string_view title, const std::unsigned_integral auto width, const std::unsigned_integral auto height, const std::integral auto x, const std::integral auto y) {

            glutInitWindowSize(width, height);
            glutInitWindowPosition(x, y);
            window_handle_ = glutCreateWindow(title.data());

            glClearColor(0.20f, 0.0f, 0.15f, 1.0f);
            glutSetWindow(window_handle_); // TODO: is this call needed?
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0.0f, width - 1, 0, height - 1, -1.0f, 1.0f);
            glMatrixMode(GL_MODELVIEW);

            //glViewport(x, y, width, height);
            //std::cout << "glViewport: error=" << glGetError() << "\n";
            

            std::function<void()> close_func = [&]() { window_is_open_ = false; };

            //glutCloseFunc(close_func); // TODO: implement c-style callback, unfortunately this doesn't work though since there's no context in the call to close_func()
        }

        ~Window() {
            if (window_is_open_) {
                std::cout << "window::~window: destroying window...\n";
                //glutDestroyWindow(window_handle_); // TODO: implement close callback,
                // TODO: or just call glutDestroyWindow() and disregard returned error? probably disregard, doesn't seem to cause any problems to just disregard returned error
            }
            std::cout << "window::~window: destructor finalized\n";
        }

        void clear() {
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        }

        void switch_to() {
            glutSetWindow(window_handle_);
        }

        bool has_focus() {
            return glutGetWindow() == window_handle_;
        }

        void dispatch_events() {
            glutMainLoopEvent();
        }

        void swap() {
            glutSwapBuffers();
        }

        void resize(const std::unsigned_integral auto width, const std::unsigned_integral auto height) {
            glutReshapeWindow(width, height);
        }

        void move(const std::integral auto x, const std::integral auto y) {
            glutPositionWindow(x, y);
        }

        void fullscreen(const bool enabled=true) {
            if (enabled)
                glutFullScreen();
            else
                glutLeaveFullScreen();
        }

        bool is_open() { return window_is_open_; }

    private:
        int window_handle_{0};
        bool window_is_open_{true};
};

template<std::unsigned_integral T=unsigned int>
class Viewer {
    public:
        Viewer(std::string_view title, const std::unsigned_integral auto width, const std::unsigned_integral auto height) : width_(width), height_(height), window_(Window(title, width, height)) {
            //display_ = get_display();
            //window_ = get_window(width, height, display_);
            //window_.on_close = window_close_by_user; // TODO: implement
            //window_is_open_ = true;

            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }

        ////void set_bounds(const std::unsigned_integral auto left, const std::unsigned_integral auto right, const std::unsigned_integral auto bottom, const std::unsigned_integral auto top) {
        //void set_bounds(const std::integral auto left, const std::integral auto right, const std::integral auto bottom, const std::integral auto top) {
        void set_bounds(const std::floating_point auto left, const std::floating_point auto right, const std::floating_point auto bottom, const std::floating_point auto top) {
            //assert(right > left && top > bottom); // TODO: assert

            auto scale_x = width_ / (right - left);
            auto scale_y = height_ / (top - bottom);

            transform_.set_translation(point_2d{-left * scale_x, -bottom * scale_y});
            transform_.set_scale(point_2d{scale_x, scale_y});
        }

        /*void add_geom(geom_concept auto geom) {
            geoms_.push_back(std::move(geom));
        }*/

        void add_geom(geom_concept auto& geom) {
            geoms_.push_back(geom);
        }

        /*void add_onetime(geom_concept auto geom) {
            onetime_geoms_.push_back(std::move(geom));
        }*/

        void add_onetime(geom_concept auto& geom) {
            onetime_geoms_.push_back(geom);
        }

        template<bool return_rgb_array=false>
        auto render() {
            //std::cout << "begin: viewer.render\n"; // TODO: add as debug info?
            glClearColor(0.15f, 0.0f, 0.1f, 1.0f);
            window_.switch_to();
            window_.clear(); // TODO: glut standard says call switch_to() before clear()
            window_.dispatch_events();
            transform_.enable();

            for (auto& geom : geoms_) {
                ////std::cout << " begin: viewer.render: geom=" << geom.get().to_string() << "\n";
                geom.get().render();
                ////std::cout << " end: viewer.render: geom=" << geom.get().to_string() << "\n";
            }

            for (auto& onetime_geom : onetime_geoms_) {
                ////std::cout << "onetime_geom: " << onetime_geom.get().to_string() << '\n';
                onetime_geom.get().render();
            }

            transform_.disable();

            onetime_geoms_.clear();

            //std::cout << "end: viewer.render\n"; // TODO: add as debug info?

            if constexpr (return_rgb_array) { // TODO: test saving returned rgb array to file
                std::vector<uint8_t> buffer(width_ * height_ * 4);
                glReadBuffer(GL_BACK);
                glReadPixels(0, 0, width_, height_, GL_BGRA, GL_UNSIGNED_BYTE, &buffer[0]);
                window_.swap();
                return buffer;

            } else {
                window_.swap();
                return window_.is_open(); //window_is_open_;
            }
        }

        /*template<typename T=FilledPolygon>
        T draw_polygon(vertices_2d vertices, const Color& color) {
            auto geom = make_polygon<T>(std::move(vertices));

            const auto& [red, green, blue] = color.get_color();
            geom.set_color(red, green, blue);

            add_onetime();
        }*/

        /*virtual ~Viewer() {
            //window_.close();
            //window_is_open_ = false;
        }*/

    protected:
        struct display_initializer {
            explicit display_initializer(int argc, char** argv,
                std::unsigned_integral auto width,
                    std::unsigned_integral auto height,
                        std::integral auto x, std::integral auto y)
            {
                init_display(argc, argv, width, height, x, y);
            }
        };

        /*uint64_t get_display() {
            return 7;
        }*/

        /*uint64_t get_window(T width, T height, uint64_t display) {
            return 9;
        }*/

        /*void window_closed_by_user() {
            window_is_open_ = false;
        }*/

    private:
        T width_{};
        T height_{};
        //uint64_t display_;
        display_initializer display_initializer_{0, nullptr, width_, height_, 0, 0};
        Window window_;
        //bool window_is_open_{false};
        std::vector<std::reference_wrapper<Geom>> geoms_;
        std::vector<std::reference_wrapper<Geom>> onetime_geoms_;
        Transform transform_;
};

}
