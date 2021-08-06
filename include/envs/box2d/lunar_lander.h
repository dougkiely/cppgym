#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <limits>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

#include <Box2D/Box2D.h>

#include "spaces/box.h"
#include "spaces/discrete.h"
#include "envs/classic_control/rendering.h"

using namespace gym::envs::classic_control;

namespace gym::envs::box2d {

// TODO: implement noexcept clauses

template<std::floating_point T=float>
class lunar_lander_window;

/*template<typename T>
concept lunar_lander_scene = requires(T& t, std::size_t ) {
    { t.sky_polys } -> std::same_as<std::array<std::array<float, 2>, 4>>;
    { t.particles } -> std::
};*/

template<std::floating_point T=float, bool continuous=false>
class lunar_lander {
    public:
        static const uint64_t action_size{continuous ? 2 : 4};
        static const uint64_t observation_size{8};

        using action_space_type = std::conditional_t<continuous,
                                                     gym::spaces::box<T, action_size>,
                                                     gym::spaces::discrete<>>;
        using action_type = action_space_type::value_type;
        using observation_space_type = gym::spaces::box<T, observation_size>;
        using observation_type = observation_space_type::value_type;
        using reward_type = T;
        using step_return_type = std::tuple<observation_type, reward_type, bool>;

        // useful range is -1 .. +1, but spikes can be higher
        //observation_space_type observation_space{std::numeric_limits<T>::lowest(),
        //                                         std::numeric_limits<T>::max()};
        observation_space_type observation_space{-std::numeric_limits<T>::infinity(),
                                                  std::numeric_limits<T>::infinity()};

        static constexpr auto action_values = [act_size=action_size]() -> auto {
            if constexpr (continuous)
                /* action is two floats [main engine, left-right engines].
                 * main engine: -1.0..0.0 off,
                 *               0.0..1.0 throttle from 50% to 100% power.
                 *                engine can't work with less than 50% power.
                 * left-right:  -1.0..-0.5 fire left engine,
                 *              +0.5..+1.0 fire right engine,
                 *              -0.5..+0.5 off
                 */
                return action_space_type(action_type{{-1}, {1}});
            else
                // Nop, fire left engine, main engine, right engine
                return action_space_type(act_size);
        };

        action_space_type action_space{action_values()};

        lunar_lander() {

            //seed();

            world_.SetContactListener(&contact_listener_);
            //world_.SetDestructionListener(&destruction_listener_);

            reset();
        }

        //seed() {}

        auto step(const action_type& act) -> step_return_type {

            auto action = act;

            if constexpr (continuous)
                action = std::clamp(action, -1.0f, 1.0f);
            else
                assert(action_space.contains(action));
                //if (!action_space.contains(action))
                //    return {observation_type{0, 0, 0, 0, 0, 0, 0, 0}, 0.0, true};

            auto& lander = lander_.body;

            // engines
            const auto lander_angle = lander->GetAngle();
            const auto tip = std::array{std::sin(lander_angle), std::cos(lander_angle)};
            const auto side = std::array{-tip[1], tip[0]};
            const auto dispersion = std::array{engine_dispersion_dist_(rand_gen_)/SCALE_,
                                               engine_dispersion_dist_(rand_gen_)/SCALE_};

            T m_power{0.0};

            bool is_valid_action{false};
            if constexpr (continuous)
                is_valid_action = action[0] > 0.0;
            else
                is_valid_action = action == 2;

            if (is_valid_action) {

                // main engine
                if constexpr (continuous) {
                    m_power = (std::clamp(action[0], 0.0, 1.0) + 1.0) * 0.5; // 0.5..1.0
                    assert(m_power >= 0.5 && m_power <= 1.0);
                } else {
                    m_power = 1.0;
                }

                const auto ox = (tip[0] * (4/SCALE_ + 2 * dispersion[0]) +
                                 side[0] * dispersion[1]); // 4 is move a bit downwards, +-2 for randomness
                const auto oy = -tip[1] * (4/SCALE_ + 2 * dispersion[0]) - side[1] * dispersion[1];
                const auto& lander_position = lander->GetPosition();
                b2Vec2 impulse_pos{lander_position(0) + ox, lander_position(1) + oy};

                auto p = create_particle(3.5, // 3.5 to make particle speed adequate
                                         impulse_pos(0), impulse_pos(1),
                                         m_power); // particles are just a decoration

                b2Vec2 impulse{ox * MAIN_ENGINE_POWER_ * m_power,
                               oy * MAIN_ENGINE_POWER_ * m_power};

                p->ApplyLinearImpulse(impulse, impulse_pos, true);

                impulse.x = -impulse.x;
                impulse.y = -impulse.y;

                lander->ApplyLinearImpulse(impulse, impulse_pos, true);
            }

            T s_power{0.0};

            if constexpr (continuous)
                is_valid_action = std::abs(action[1]) > 0.5;
            else
                is_valid_action = action == 1 || action == 3;

            if (is_valid_action) {

                using direction_type = std::conditional_t<continuous, T, action_type>;

                direction_type direction{};

                // orientation engines
                if constexpr (continuous) {
                    direction = action[1] == 0 ? 0 : (std::signbit(action[1]) ? -1 : 1);
                    s_power = std::clamp(std::abs(action[1]), 0.5, 1.0);
                    assert(s_power >= 0.5 && s_power <= 1.0);
                } else {
                    direction = action - 2;
                    s_power = 1.0;
                }

                const auto ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY_SCALED_);
                const auto oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY_SCALED_);

                const auto& lander_position = lander->GetPosition();
                const b2Vec2 impulse_pos{lander_position(0) + ox - tip[0] * 17/SCALE_,
                                         lander_position(1) + oy + tip[1] * SIDE_ENGINE_HEIGHT_SCALED_};

                auto p = create_particle(0.7, impulse_pos(0), impulse_pos(1), s_power);

                b2Vec2 impulse{ox * SIDE_ENGINE_POWER_ * s_power,
                               oy * SIDE_ENGINE_POWER_ * s_power};

                p->ApplyLinearImpulse(impulse, impulse_pos, true);

                impulse.x = -impulse.x;
                impulse.y = -impulse.y;

                lander->ApplyLinearImpulse(impulse, impulse_pos, true);
            }

            constexpr auto time_step{1.0f / FPS_};
            constexpr auto velocity_iterations{6 * 30};
            constexpr auto position_iterations{2 * 30};

            world_.Step(time_step, velocity_iterations, position_iterations);

            const auto& position = lander->GetPosition();
            const auto& velocity = lander->GetLinearVelocity();

            constexpr auto W_SCALED_HALF{W_ / 2};
            constexpr auto H_SCALED_HALF{H_ / 2};
            constexpr auto W_SCALED_HALF_FPS{W_SCALED_HALF / FPS_};
            constexpr auto H_SCALED_HALF_FPS{H_SCALED_HALF / FPS_};

            observation_type state{
                        (position.x - W_SCALED_HALF) / H_SCALED_HALF,
                         position.y - (helipad_y_ + LEG_DOWN_SCALED_) / H_SCALED_HALF,
                         velocity.x * W_SCALED_HALF_FPS,
                         velocity.y * H_SCALED_HALF_FPS,
                         lander->GetAngle(),
                         20.0f * lander->GetAngularVelocity() / FPS_,
                         lander_.legs[0].ground_contact ? 1.0f : 0.0f,
                         lander_.legs[1].ground_contact ? 1.0f : 0.0f};

            reward_type reward{0};
            const auto shaping = 100 * std::sqrt(state[0]*state[0] + state[1]*state[1])
                               - 100 * std::sqrt(state[2]*state[2] + state[3]*state[3])
                               - 100 * std::abs(state[4]) + 10*state[6] + 10*state[7]; // And
                         // ten points for legs contact, the idea is if you lose
                         // contact again after landing, you get negative reward

            if (prev_shaping_)
                reward = shaping - prev_shaping_;
            prev_shaping_ = shaping;

            reward -= m_power*0.30; // less fuel spent is better, about -30 for heuristic landing
            reward -= s_power*0.30;

            bool done{false};

            if (game_over_ || std::abs(state[0]) >= 1.0) {
                done = true;
                reward = -100;
            }

            if (!lander->IsAwake()) {
                done = true;
                reward = 100;
            }

            return {state, reward, done};
        }

        auto reset() -> observation_type {
            // don't call destroy(), call clean_particles(true) instead,
            // because we're not destroying the moon and lander each time
            // (just re-positioning moon and lander to initial coordinates)
            clean_particles(true);

            game_over_ = false;
            prev_shaping_ = {};

            create_terrain();

            lander_.reset();

            if constexpr (continuous)
                return std::get<0>(step({0, 0}));
            else
                return std::get<0>(step(0));
        }

        enum class render_mode {
            human,
            rgb_array
        };

        template<render_mode mode=render_mode::human>
        auto render() {
            static lunar_lander_window<T> window(VIEWPORT_W_, VIEWPORT_H_);//, SCALE_);
            return window.template render<mode>(this, helipad_x1_, helipad_x2_);
        }

        void close() {}

        virtual ~lunar_lander() {
            destroy();
        }

    protected:
        void create_terrain() {

            std::array<T, CHUNKS_+1> height;

            for (auto& h : height)
                h = terrain_height_dist_(rand_gen_);

            std::array<T, CHUNKS_> chunk_x;

            for (std::size_t i=0; i < chunk_x.size(); ++i)
                chunk_x[i] = W_ / (CHUNKS_-1) * i;

            helipad_x1_ = chunk_x[static_cast<std::size_t>(std::floor(CHUNKS_ / 2-1))];
            helipad_x2_ = chunk_x[static_cast<std::size_t>(std::floor(CHUNKS_ / 2+1))];

            height[static_cast<std::size_t>(CHUNKS_ / 2-2)] = helipad_y_;
            height[static_cast<std::size_t>(CHUNKS_ / 2-1)] = helipad_y_;
            height[static_cast<std::size_t>(CHUNKS_ / 2+0)] = helipad_y_;
            height[static_cast<std::size_t>(CHUNKS_ / 2+1)] = helipad_y_;
            height[static_cast<std::size_t>(CHUNKS_ / 2+2)] = helipad_y_;

            std::array<T, CHUNKS_> smooth_y;

            for (size_t i=0; i < smooth_y.size(); ++i) {
                size_t h0 = i == 0 ? smooth_y.size() - 1 : i-1;
                size_t h1 = i;
                size_t h2 = i+1;
                    
                smooth_y[i] = 0.33 * (height[h0] + height[h1] + height[h2]);
            }

            for (size_t i=0; i < CHUNKS_-1; ++i) {
                b2Vec2 p1{chunk_x[i], smooth_y[i]};
                b2Vec2 p2{chunk_x[i+1], smooth_y[i+1]};

                b2FixtureDef fixture_def;

                b2EdgeShape edge_shape;
                edge_shape.m_vertex1 = std::move(p1);//edge_shape.SetTwoSided(p1, p2);
                edge_shape.m_vertex2 = std::move(p2);

                fixture_def.shape = &edge_shape;
                fixture_def.density = 0;
                fixture_def.friction = DEFAULT_FRICTION_;

                moon_.body->CreateFixture(&fixture_def);

                sky_polys_[i] = {std::array{p1(0), p1(1)},
                                 std::array{p2(0), p2(1)},
                                 std::array{p2(0), H_},
                                 std::array{p1(0), H_}};
            }
        }

        b2Body* create_particle(const T mass, const T x, const T y, const T ttl) {

            b2BodyDef body_def;
            body_def.type = b2_dynamicBody;
            body_def.position.Set(x, y);
            body_def.angle = 0.0;

            auto body = world_.CreateBody(&body_def);
            {
                b2FixtureDef fixture_def;

                b2CircleShape circle_shape;

                fixture_def.shape = &circle_shape;
                fixture_def.density = mass;
                fixture_def.friction = DEFAULT_FRICTION_;
                fixture_def.filter.categoryBits = PARTICLE_COLLISION_CATEGORY_;
                fixture_def.restitution = 0.3;

                body->CreateFixture(&fixture_def); // fixture will be destroyed with body
            }

            particles_.emplace_back(body, ttl);
            clean_particles(false);

            return body;
        }

        void clean_particles(const bool all) {
            for (const auto& particle : particles_)
                if (all || particle.ttl < 0)
                    world_.DestroyBody(particle.body);

            particles_.clear();
        }

        void destroy() {
            world_.SetContactListener(nullptr);

            clean_particles(true);

            moon_.destroy();
            lander_.destroy();
        }

    private:
        static constexpr unsigned int FPS_{50};
        static constexpr T SCALE_{30.0}; // affects how fast-paced the game is, forces should be adjusted as well

        const T MAIN_ENGINE_POWER_{13.0};
        const T SIDE_ENGINE_POWER_{0.6};

        static constexpr T INITIAL_RANDOM_{1000.0}; // set 1500 to make game harder

        static constexpr std::array<std::array<T, 2>, 6> LANDER_POLY_{
                                               std::array{-14.0f,  17.0f},
                                               std::array{-17.0f,   0.0f},
                                               std::array{-17.0f, -10.0f},
                                               std::array{ 17.0f, -10.0f},
                                               std::array{ 17.0f,   0.0f},
                                               std::array{ 14.0f,  17.0f}};

        static constexpr T LEG_AWAY_{20};
        static constexpr T LEG_DOWN_{18};
        static constexpr T LEG_W_{2};
        static constexpr T LEG_H_{8};
        static constexpr T LEG_SPRING_TORQUE_{40};

        static constexpr T LEG_AWAY_SCALED_{LEG_AWAY_/SCALE_};
        static constexpr T LEG_DOWN_SCALED_{LEG_DOWN_/SCALE_};
        static constexpr T LEG_W_SCALED_{LEG_W_/SCALE_};
        static constexpr T LEG_H_SCALED_{LEG_H_/SCALE_};

        static constexpr T SIDE_ENGINE_HEIGHT_{14.0};
        static constexpr T SIDE_ENGINE_HEIGHT_SCALED_{SIDE_ENGINE_HEIGHT_/SCALE_};
        static constexpr T SIDE_ENGINE_AWAY_{12.0};
        static constexpr T SIDE_ENGINE_AWAY_SCALED_{SIDE_ENGINE_AWAY_/SCALE_};

        static constexpr unsigned int VIEWPORT_W_{600};
        static constexpr unsigned int VIEWPORT_H_{400};

        static constexpr T W_{VIEWPORT_W_ / SCALE_};
        static constexpr T H_{VIEWPORT_H_ / SCALE_};

        static constexpr unsigned int CHUNKS_{11};

        static constexpr T INITIAL_X_{W_ / 2};
        static constexpr T INITIAL_Y_{H_ / 2}; //{H_};

        static constexpr T DEFAULT_FRICTION_{0.1};
        static constexpr uint16_t LANDER_COLLISION_CATEGORY_  {0x0010};
        static constexpr uint16_t LEG_COLLISION_CATEGORY_     {0x0020};
        static constexpr uint16_t PARTICLE_COLLISION_CATEGORY_{0x0100};
        static constexpr uint16_t GROUND_COLLISION_MASK_       {0x001}; // collide only with ground

        std::mt19937 rand_gen_{1};
        std::uniform_real_distribution<T> terrain_height_dist_{0.0, H_ / 2};
        std::uniform_real_distribution<T> engine_dispersion_dist_{-1.0, 1.0};

        template<std::floating_point color_type=T>
        struct Color {
            Color(color_type r, color_type g, color_type b) : red(r), green(g), blue(b) {}
            Color()=default;
            Color(const Color&)=default;
            Color& operator=(const Color&)=default;
            Color(Color&&)=default;
            Color& operator=(Color&&)=default;
            color_type red{0.0};
            color_type green{0.0};
            color_type blue{0.0};
        };

        using SkyPolygon = std::array<std::array<T, 2>, 4>;

        using SkyTerrain = std::array<SkyPolygon, CHUNKS_-1>;

        struct Moon {
            b2Body* body{nullptr};
            Color<T> color1{1.0, 1.0, 1.0}; //{0.0, 0.0, 0.0};
            Color<T> color2{1.0, 1.0, 1.0}; //{0.0, 0.0, 0.0};

            Moon(b2World& world) {
                b2BodyDef body_def;
                body_def.type = b2_staticBody;

                body = world.CreateBody(&body_def);
                {
                    b2FixtureDef fixture_def;

                    b2EdgeShape edge_shape;
                    edge_shape.m_vertex1 = {0.0, 0.0};
                    edge_shape.m_vertex2 = {W_, 0.0};

                    fixture_def.shape = &edge_shape;

                    body->CreateFixture(&fixture_def);
                }
            }

            void destroy() {
                if (body != nullptr) {
                    body->GetWorld()->DestroyBody(body);
                    body = nullptr; // TODO: use const_cast<> or remove_const<>?
                }
            }
         };

        struct LanderLeg {
            using leg_direction_type = int;
            static const int LEFT_LEG{-1};
            static const int RIGHT_LEG{1};

            b2Body* body{nullptr};
            b2Joint* joint{nullptr};
            bool ground_contact{false};
            Color<T> color1{0.5, 0.4, 0.9};
            Color<T> color2{0.3, 0.3, 0.5};

            LanderLeg()=default;
            LanderLeg(const LanderLeg& other)=delete;
            LanderLeg& operator=(const LanderLeg& other)=delete;
            LanderLeg(LanderLeg&& other)=default;
            LanderLeg& operator=(LanderLeg&& other)=default;

            LanderLeg(/*b2World& world, */b2Body *const lander, const leg_direction_type leg_x_direction) :
                leg_x_direction_(leg_x_direction),
                default_x_position_(INITIAL_X_ - leg_x_direction_*LEG_AWAY_/SCALE_),
                default_y_position_(INITIAL_Y_),
                default_angle_(leg_x_direction_*0.05)
             {
                b2BodyDef body_def;
                body_def.type = b2_dynamicBody;
                body_def.position.Set(default_x_position_, default_y_position_);
                body_def.angle = default_angle_;

                body = lander->GetWorld()->CreateBody(&body_def);
                //body = world.CreateBody(&body_def);
                {
                    b2FixtureDef fixture_def;

                    b2PolygonShape polygon_shape;
                    polygon_shape.SetAsBox(LEG_W_SCALED_, LEG_H_SCALED_);
                    if (polygon_shape.Validate())
                        std::cout << "LANDER LEG[" << leg_x_direction << "]: THE POLYGON SHAPE IS VALID\n";
                    else
                        std::cout << "LANDER LEG[" << leg_x_direction << "]: THE POLYGON SHAPE IS **NOT** VALID\n";

                    fixture_def.shape = &polygon_shape;
                    fixture_def.density = 1.0;
                    fixture_def.restitution = 0.0; // 0.99 bouncy
                    fixture_def.filter.categoryBits = LEG_COLLISION_CATEGORY_;
                    fixture_def.filter.maskBits = GROUND_COLLISION_MASK_;

                    auto f = body->CreateFixture(&fixture_def);
                    if (f == nullptr)
                        std::cout << "LANDER LEG: fixture is nullptr\n";
                    else {
                        std::cout << "LANDER LEG: fixture is NOT nullptr\n";
                        auto shape = f->GetShape();
                        std::cout << "LANDER LEG: fixture shape is " << shape->GetType() << '\n';
                    }
                }

                b2RevoluteJointDef rjd;

                rjd.bodyA = lander;
                rjd.bodyB = body;
                rjd.localAnchorA = {0, 0};
                rjd.localAnchorB = {leg_x_direction_ * LEG_AWAY_/SCALE_, LEG_DOWN_SCALED_};
                rjd.enableMotor = true;
                rjd.enableLimit = true;
                rjd.maxMotorTorque = LEG_SPRING_TORQUE_;
                rjd.motorSpeed = 0.3 * leg_x_direction_; // low enough not to jump back into the sky

                if (leg_x_direction_ == LEFT_LEG) {
                    rjd.lowerAngle = 0.9 - 0.5; // the most esoteric numbers here, angled legs have freedom to travel within
                    rjd.upperAngle = 0.9;
                } else {
                    rjd.lowerAngle = -0.9;
                    rjd.upperAngle = -0.9 + 0.5;
                }

                joint = lander->GetWorld()->CreateJoint(&rjd);
                //joint = world.CreateJoint(&rjd);
            }

            void reset() {
                body->SetTransform({default_x_position_, default_y_position_}, default_angle_);
            }

            void destroy() {
                /*if (joint != nullptr) {
                    if (body != nullptr)
                        body->GetWorld()->DestroyJoint(joint);
                    joint = nullptr;
                }*/

                if (body != nullptr) {
                    body->GetWorld()->DestroyBody(body);
                    body = nullptr;
                    joint = nullptr;
                }
            }

            ~LanderLeg() {
                destroy();
            }

            private:
                leg_direction_type leg_x_direction_;
                float default_x_position_;
                float default_y_position_;
                float default_angle_;
        };

        struct Lander {
            using LanderLegs = std::array<LanderLeg, 2>;
            //using LanderLegs = std::vector<LanderLeg>;

            b2Body* body{nullptr};
            Color<T> color1{0.5, 0.4, 0.9};
            Color<T> color2{0.3, 0.3, 0.5};
            LanderLegs legs;

            Lander(b2World& world, std::mt19937& rand_gen) : rand_gen_(rand_gen) {
                b2BodyDef body_def;
                body_def.type = b2_dynamicBody;
                body_def.position.Set(INITIAL_X_, INITIAL_Y_);
                body_def.angle = 0.0;

                body = world.CreateBody(&body_def);
                {
                    b2FixtureDef fixture_def;

                    b2PolygonShape polygon_shape;

                    std::array<b2Vec2, LANDER_POLY_.size()> vertices;
                    for (std::size_t i=0; i < vertices.size(); ++i) {
                        vertices[i].x = LANDER_POLY_[i][0]/SCALE_;
                        vertices[i].y = LANDER_POLY_[i][1]/SCALE_;
                    }

                    polygon_shape.Set(vertices.cbegin(), vertices.size());
                    if (polygon_shape.Validate())
                        std::cout << "THE LANDER POLYGON SHAPE IS VALID\n";
                    else
                        std::cout << "THE LANDER POLYGON SHAPE IS **NOT** VALID\n";

                    fixture_def.shape = &polygon_shape;
                    fixture_def.density = 5.0;
                    fixture_def.friction = DEFAULT_FRICTION_;
                    fixture_def.filter.categoryBits = LANDER_COLLISION_CATEGORY_;
                    fixture_def.filter.maskBits = GROUND_COLLISION_MASK_;
                    fixture_def.restitution = 0.0; // 0.99 bouncy

                    body->CreateFixture(&fixture_def); // fixture will be destroyed with body
                }

                legs[0] = std::move(LanderLeg(/*world, */body, LanderLeg::LEFT_LEG));
                legs[1] = std::move(LanderLeg(/*world, */body, LanderLeg::RIGHT_LEG));
                //legs.emplace_back(world, body, -1); //LanderLeg::LEFT_LEG);
                //legs.emplace_back(world, body,  1); //LanderLeg::RIGHT_LEG);
            }

            void reset() {
                for (auto& leg : legs)
                    leg.reset();
                //legs.clear();

                const T angle{0.0};
                body->SetTransform({INITIAL_X_, INITIAL_Y_}, angle);

                body->ApplyForceToCenter({lander_force_dist_(rand_gen_),
                                          lander_force_dist_(rand_gen_)}, true);
            }

            virtual ~Lander() {
                destroy();
            }

            void destroy() {
                for (auto& leg : legs)
                    leg.destroy();
                //legs.clear();

                if (body != nullptr) {
                    body->GetWorld()->DestroyBody(body);
                    body = nullptr;
                }
            }

            protected:
                std::mt19937 rand_gen_;
                std::uniform_real_distribution<T> lander_force_dist_{-INITIAL_RANDOM_, INITIAL_RANDOM_};
        };

        template<std::floating_point ttl_value_type=T>
        struct Particle {
            //particle(b2Body* body, ttl_value_type ttl) : body(body), ttl(ttl) {}
            b2Body* body{nullptr};
            ttl_value_type ttl{};
            Color<T> color1{};
            Color<T> color2{};
        };

        class contact_detector : public b2ContactListener {
            public:
                contact_detector(lunar_lander::Lander* lander, bool& game_over) : lander_(lander), game_over_(game_over) {}

                void BeginContact(b2Contact* contact) override {
                    if (lander_->body == contact->GetFixtureA()->GetBody() || lander_->body == contact->GetFixtureB()->GetBody())
                        game_over_ = true;

                    for (auto& leg : lander_->legs)
                        if (leg.body == contact->GetFixtureA()->GetBody() || leg.body == contact->GetFixtureB()->GetBody())
                            leg.ground_contact = true;
                }

                void EndContact(b2Contact* contact) override {
                    for (auto& leg : lander_->legs)
                        if (leg.body == contact->GetFixtureA()->GetBody() || leg.body == contact->GetFixtureB()->GetBody())
                            leg.ground_contact = false;
                }

            private:
                Lander* lander_;
                bool& game_over_;
        };

        b2Vec2 gravity_{0.0f, -10.0f};

        b2World world_{gravity_};
        SkyTerrain sky_polys_; // terrain
        Moon moon_{world_};
        T helipad_x1_{};
        T helipad_x2_{};
        static constexpr T helipad_y_{H_/4};
        Lander lander_{world_, rand_gen_};
        std::vector<Particle<T>> particles_;

        contact_detector contact_listener_{&lander_, game_over_};
        //b2DestructionListener destruction_listener_;

        T prev_shaping_{};

        bool game_over_{false};

        friend lunar_lander_window<T>;
};

template<std::floating_point T>
class lunar_lander_window {
    public:
        explicit lunar_lander_window(const unsigned int width, const unsigned int height)
            : screen_width_(width), screen_height_(height) {

            viewer_.set_bounds(0.0f, screen_width_/SCALE_, 0.0f, screen_height_/SCALE_);
        }

        template<lunar_lander<T>::render_mode mode=lunar_lander<T>::render_mode::human>
        auto render(lunar_lander<T>*const env, const T helipad_x1, const T helipad_x2) {

            /*for (auto& particle : env->particles_) {
                particle.ttl -= 0.15;

                typename lunar_lander<T>::Color<T> color{
                                     std::max(0.2f, 0.2f + particle.ttl),
                                     std::max(0.2f, 0.5f * particle.ttl),
                                     std::max(0.2f, 0.5f * particle.ttl)};
 
                particle.color1 = color;
                particle.color2 = color;
            }

            env->clean_particles(false);*/

            for (std::size_t i=0; i < sky_polygons_.size(); ++i) {

                sky_polygons_[i].set_vertices({{env->sky_polys_[i][0][0],
                                                env->sky_polys_[i][0][1]},
                                               {env->sky_polys_[i][1][0],
                                                env->sky_polys_[i][1][1]},
                                               {env->sky_polys_[i][2][0],
                                                env->sky_polys_[i][2][1]},
                                               {env->sky_polys_[i][3][0],
                                                env->sky_polys_[i][3][1]}});
                //polygon[i].set_vertices(env->sky_polys_[i]);

                viewer_.add_onetime(sky_polygons_[i]);
            }

            lander_.draw(viewer_, env->lander_);
            helipad_.draw(viewer_, helipad_x1, helipad_x2);

            if (!initialized_) {
                lander_.set_color(env->lander_.color1, env->lander_.color2);
                lander_.set_line_width(2);
                initialized_ = true;
            }

            return viewer_.render<mode == lunar_lander<T>::render_mode::rgb_array>();
        }

    private:
        class LunarLander {
            public:
                void set_color(const typename lunar_lander<T>::Color<T>& color1,
                                const typename lunar_lander<T>::Color<T>& color2) {

                    body_polygon_.set_color(color1.red, color1.green, color1.blue);
                    body_polyline_.set_color(color2.red, color2.green, color2.blue);

                    for (auto& leg_polygon : legs_polygon_)
                        leg_polygon.set_color(color1.red, color1.green, color1.blue);

                    for (auto& leg_polyline : legs_polyline_)
                        leg_polyline.set_color(color2.red, color2.green, color2.blue);
                }

                void set_line_width(LineWidth::value_type line_width) {
                    body_polyline_.set_width(line_width);

                    for (auto& leg_polyline : legs_polyline_)
                        leg_polyline.set_width(line_width);
                }

                void draw(Viewer<>& viewer, lunar_lander<T>::Lander& lander) {
                    std::cout << "enter: LunarLander::draw(Lander): " << '\n';

                    draw(viewer, lander.body, body_polygon_, body_polyline_);

                    for (std::size_t i=0; i < lander.legs.size(); ++i)
                        draw(viewer, lander.legs[i].body, legs_polygon_[i], legs_polyline_[i]);

                    std::cout << "exit: LunarLander::draw(Lander): " << '\n';
                }

                void draw(Viewer<>& viewer, b2Body*const body, FilledPolygon& body_polygon, PolyLine& body_polyline) {
                    std::cout << "enter: LunarLander::draw(b2Body): " << '\n';

                    auto fixture = body->GetFixtureList();

                    size_t fixture_count{};

                    while (fixture != nullptr) {

                        std::cout << "Fixture count: " << ++fixture_count << '\n';

                        const auto& tf = fixture->GetBody()->GetTransform();
                        // tf.p: b2Vec2
                        //    p.x, p.y
                        // tf.q: b2Rot
                        //    q.s, q.c

                        auto shape_tmp = fixture->GetShape();

                        std::cout << "SHAPE TMP TYPE: " << shape_tmp->GetType() << ", childCount: " << shape_tmp->GetChildCount() << '\n';

                        const auto *const shape = static_cast<b2PolygonShape*>(fixture->GetShape());

                        std::cout << "SHAPE TYPE: " << shape->GetType() << '\n';

                        if (shape == nullptr) {
                            std::cout << "SHAPE IS NOT A POLYGON\n";
                            return;
                        } else {
                            std::string shape_str;
                            switch (shape->GetType()) {
                                case 0: shape_str = "circle"; break;
                                case 1: shape_str = "edge"; break;
                                case 2: shape_str = "polygon"; break;
                                default: shape_str = "unknown";
                            }
                            std::cout << "SHAPE IS A " << shape_str << '\n';;
                        }

                        const auto& count = shape->m_count;
                        std::cout << "shape->m_count: " << shape->m_count << '\n';
                        const auto& shape_vertices = shape->m_vertices;

                        std::vector<FilledPolygon::value_type> vertices;

                        for (int i=0; i < count; ++i) {

                            const auto& v = shape_vertices[i];

                            const auto x = (tf.q.c * v.x - tf.q.s * v.y) + tf.p.x;
                            const auto y = (tf.q.s * v.x + tf.q.c * v.y) + tf.p.y;

                            vertices.emplace_back(x, y);
                        }

                        for (const auto& point : vertices)
                            std::cout << "vertice point: " << point.x << ", " << point.y << '\n';

                        auto vertices1 = vertices;
                        if (!vertices1.empty())
                            vertices1.push_back(vertices1[0]);

                        body_polygon.set_vertices(vertices);
                        body_polyline.set_vertices(vertices1);

                        viewer.add_onetime(body_polygon);
                        viewer.add_onetime(body_polyline);

                        fixture = fixture->GetNext();
                    }

                    std::cout << "exit: LunarLander::draw(b2Body): " << '\n';
                }

                void draw_v1(Viewer<>& viewer, lunar_lander<T>::Lander& lander) {
                    std::cout << "enter: LunarLander::draw(): " << '\n';

                    auto fixture = lander.body->GetFixtureList();

                    size_t fixture_count{};

                    while (fixture != nullptr) {

                        std::cout << "Fixture count: " << ++fixture_count << '\n';

                        const auto& tf = fixture->GetBody()->GetTransform();
                        // tf.p: b2Vec2
                        //    p.x, p.y
                        // tf.q: b2Rot
                        //    q.s, q.c

                        const auto *const shape = static_cast<b2PolygonShape*>(fixture->GetShape());

                        const auto& count = shape->m_count;
                        std::cout << "shape->m_count: " << shape->m_count << '\n';
                        const auto& shape_vertices = shape->m_vertices;

                        std::vector<FilledPolygon::value_type> vertices;

                        for (int i=0; i < count; ++i) {

                            const auto& v = shape_vertices[i];

                            const auto x = (tf.q.c * v.x - tf.q.s * v.y) + tf.p.x;
                            const auto y = (tf.q.s * v.x + tf.q.c * v.y) + tf.p.y;

                            vertices.emplace_back(x, y);
                        }

                        for (const auto& point : vertices)
                            std::cout << "vertice point: " << point.x << ", " << point.y << '\n';

                        //auto vertices1 = vertices;
                        //if (!vertices1.empty())
                        //    vertices1.push_back(vertices1[0]);

                        body_polygon_.set_vertices(vertices);
                        //body_polyline_.set_vertices(vertices1);

                        viewer.add_onetime(body_polygon_);
                        //viewer.add_onetime(body_polyline_);

                        fixture = fixture->GetNext();
                    }

                    std::cout << "exit: LunarLander::draw(): " << '\n';
                }

                /*void draw(Viewer<>& viewer, lunar_lander<T>::Lander& lander) {
                    std::cout << "enter: LunarLander::draw(): " << '\n';

                    auto [polygon_vertices, polyline_vertices] = get_vertices(lander.body);
                    std::cout << "exit: LunarLander::draw(): " << '\n';
                }*/

                /*void draw(Viewer<>& viewer, std::initializer_list<point_2d> vertices) {

                    body_polygon_.set_vertices(std::vector{vertices});

                    std::vector body_vec{vertices};
                    if (!body_vec.empty())
                        body_vec.push_back(body_vec[0]);
                    body_polyline_.set_vertices(body_vec);

                    for (leg_polyline_
                }*/
            protected:
                /*void get_vertices(Viewer<>& viewer, const b2Body *const body) {

                    auto fixture = body->GetFixtureList();

                    size_t fixture_count{};

                    while (fixture != nullptr) {

                        std::cout << "Fixture count: " << ++fixture_count << '\n';

                        const auto& tf = fixture->GetBody()->GetTransform();
                        // tf.p: b2Vec2
                        // tf.q: b2Rot

                        const auto *const shape = static_cast<b2PolygonShape*>(fixture->GetShape());

                        const auto& count = shape->m_count;
                        std::cout << "shape->m_count: " << shape->m_count << '\n';
                        const auto& shape_vertices = shape->m_vertices;

                        std::vector<FilledPolygon::value_type> vertices;

                        for (int i=0; i < count; ++i) {

                            const auto& point = shape_vertices[i];

                            vertices.emplace_back(tf.p.x + point.x,
                                                  tf.p.y + point.y);
                        }

                        for (const auto& point : vertices)
                            std::cout << "vertice point: " << point.x << ", " << point.y << '\n';

                        //auto vertices1 = vertices;
                        //if (!vertices1.empty())
                        //    vertices1.push_back(vertices1[0]);

                        body_polygon_.set_vertices(vertices);
                        //body_polyline_.set_vertices(vertices1);

                        viewer.add_onetime(body_polygon_);
                        //viewer.add_onetime(body_polyline_);

                        fixture = fixture->GetNext();
                    }
                }*/

            private:
                FilledPolygon body_polygon_;
                PolyLine body_polyline_;
                std::array<FilledPolygon, 2> legs_polygon_;
                std::array<PolyLine, 2> legs_polyline_;
        };

        class Helipad {
            public:
                Helipad() {
                    for (auto& flag_pole : flag_poles_)
                        flag_pole.set_color(1.0, 1.0, 1.0);

                    for (auto& flag : flags_)
                        flag.set_color(0.8, 0.8, 0.0);
                }

                void draw(Viewer<>& viewer, const T helipad_x1, const T helipad_x2) {

                    std::array helipad_x{helipad_x1, helipad_x2};

                    for (std::size_t i=0; i < helipad_x.size(); ++i) {

                        auto& x = helipad_x[i];
                        auto& flag_pole = flag_poles_[i];
                        auto& flag = flags_[i];

                        flag_pole.set_vertices({{x, flag_y1_}, {x, flag_y2_}});

                        viewer.add_onetime(flag_pole);

                        flag.set_vertices({{x, flag_y2_},
                                           {x, flag_y2_ - SCALE_10_},
                                           {x + SCALE_25_, flag_y2_ - SCALE_5_}});

                        viewer.add_onetime(flag);
                    }
                }

            private:
                static constexpr T SCALE_{lunar_lander_window::SCALE_};

                static constexpr T SCALE_5_ { 5/SCALE_};
                static constexpr T SCALE_10_{10/SCALE_};
                static constexpr T SCALE_25_{25/SCALE_};
                static constexpr T SCALE_50_{50/SCALE_};

                static constexpr T flag_y1_{lunar_lander_window::helipad_y_};
                static constexpr T flag_y2_{flag_y1_ + 50/SCALE_};

                std::array<PolyLine, 2> flag_poles_;
                std::array<FilledPolygon, 2> flags_;
        };

        const unsigned int screen_width_;
        const unsigned int screen_height_;

        static constexpr T SCALE_{lunar_lander<T>::SCALE_};
        static constexpr T helipad_y_{lunar_lander<T>::helipad_y_};

        bool initialized_{false};

        LunarLander lander_;
        Helipad helipad_;//{SCALE_, lunar_lander<T>::helipad_y_};
        std::array<FilledPolygon, lunar_lander<T>::CHUNKS_-1> sky_polygons_;

        Viewer<> viewer_{"C++ Gym - Lunar Lander", screen_width_, screen_height_};
};

}
